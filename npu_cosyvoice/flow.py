"""Replace CosyVoice3's flow-matching DiT estimator with the NPU library.

All 22 transformer blocks run in libdit_npu (int8 GEMMs on the XDNA1 NPU, the rest in C++ on the
host); time/input embedding and the output projection stay in torch. The torch blocks are dropped
after the NPU weights are loaded.
"""
import ctypes
import gc
import os
import threading
import time

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_LIB = os.path.join(HERE, "libdit_npu.so")


class NpuEstimator:
    def __init__(self, est, weights_dir, aot_dir, lib=DEFAULT_LIB, threads=4, maxrows=1536, gate=None, retries=2):
        self.est = est
        self.lib = ctypes.CDLL(lib)
        P = ctypes.c_void_p
        self.lib.dit_init.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int, ctypes.c_int, ctypes.c_int]
        self.lib.dit_forward_chunked.argtypes = [P, ctypes.c_int, P, P, P, ctypes.c_int]
        self.lib.dit_set_gate.argtypes = [ctypes.c_char_p]
        self.lib.dit_paused_seconds.restype = ctypes.c_double
        nl = len(est.transformer_blocks)
        n = self.lib.dit_init(aot_dir.encode(), weights_dir.encode(), nl, threads, maxrows)
        if n != nl:
            raise RuntimeError(f"libdit_npu init failed ({n}); see stderr")
        if gate and self.lib.dit_set_gate(gate.encode()) != 0:
            raise RuntimeError(f"cannot open NPU gate {gate}")
        self.ada_w = os.path.join(weights_dir, "adaln_w_f16.npy")
        self.ada_b = os.path.join(weights_dir, "adaln_b_f32.npy")
        est.transformer_blocks = torch.nn.ModuleList()
        gc.collect()
        try:
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except OSError:
            pass
        self.maxrows = maxrows
        self.retries = retries
        self.mod_cache = {}
        self.lock = threading.Lock()  # libdit_npu keeps its state in globals: one forward at a time
        self.calls = 0
        self.errors = 0

    def paused_seconds(self):
        return self.lib.dit_paused_seconds()

    def mods(self, t_emb, key):
        # AdaLayerNormZero modulation for every block; depends only on the timestep, so it is cached.
        m = self.mod_cache.get(key)
        if m is None:
            with torch.autocast("cpu", enabled=False):
                st = F.silu(t_emb.float())
                W = np.load(self.ada_w, mmap_mode="r")
                B = np.load(self.ada_b)
                m = torch.stack([F.linear(st, torch.from_numpy(np.asarray(W[i], dtype=np.float32)), torch.from_numpy(B[i]))
                                 for i in range(W.shape[0])]).float().contiguous()
            self.mod_cache[key] = m
        return m

    @torch.no_grad()
    def forward(self, x, mask, mu, t, spks=None, cond=None, streaming=False):
        est = self.est
        if not bool(mask.bool().all()):
            raise ValueError("the NPU path supports unpadded sequences only")
        chunk = int(est.static_chunk_size) if streaming else 0
        x = x.transpose(1, 2)
        mu = mu.transpose(1, 2)
        cond = cond.transpose(1, 2)
        batch, seq = x.shape[0], x.shape[1]
        if batch != 2 or seq > self.maxrows:
            raise ValueError(f"expected CFG batch 2 and <= {self.maxrows} frames, got {batch} x {seq}")
        if t.ndim == 0:
            t = t.repeat(batch)
        temb = est.time_embed(t)
        h = est.input_embed(x, cond, mu, spks).float().contiguous()
        freqs = est.rotary_embed.forward_from_seq_len(seq)[0][0, -seq:, :].float()
        cos = freqs.cos().contiguous()
        sin = freqs.sin().contiguous()
        mod = self.mods(temb, tuple(round(float(v), 7) for v in t.float().tolist()))
        with self.lock:
            for attempt in range(self.retries + 1):  # libdit_npu writes h only on success
                rc = self.lib.dit_forward_chunked(ctypes.c_void_p(h.data_ptr()), seq, ctypes.c_void_p(mod.data_ptr()),
                                                  ctypes.c_void_p(cos.data_ptr()), ctypes.c_void_p(sin.data_ptr()), chunk)
                if rc == 0:
                    break
                self.errors += 1
                if attempt == self.retries:
                    raise RuntimeError(f"NPU DiT forward failed ({rc}) after {attempt + 1} attempts")
                time.sleep(0.2 * (attempt + 1))
            self.calls += 1
        h = h.to(temb.dtype)
        h = est.norm_out(h, temb)
        return est.proj_out(h).transpose(1, 2)


def install(model, weights_dir, aot_dir, **kw):
    """Swap the estimator of a loaded CosyVoice3 model (cosyvoice.cli.cosyvoice.CosyVoice3)."""
    est = model.model.flow.decoder.estimator
    ne = NpuEstimator(est, weights_dir, aot_dir, **kw)
    est.forward = ne.forward
    return ne
