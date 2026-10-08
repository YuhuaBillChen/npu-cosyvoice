"""Run CosyVoice3's LM backbone (Qwen2, 0.5B) through llama.cpp on the CPU.

CosyVoice drives the backbone one step at a time: the whole prompt first, then one speech token
per step. libcv_llm (csrc/cv_llm.c) exposes exactly that over libllama; CosyVoice keeps its own
embeddings, speech-token head and sampling in torch. With a Q4_0 GGUF this decodes several times
faster than torch bf16 and reads ~0.6 of Q8_0's bytes per token, which matters when the memory
bus is shared with another model.

Prefill is compute-bound and decode is bandwidth-bound, so prefill gets more threads.

Export the GGUF once (needs the llama.cpp checkout from `make llm` for its converter and
llama-quantize):

    python -m npu_cosyvoice.llm --model-dir pretrained_models/Fun-CosyVoice3-0.5B-2512 \
        --llama-cpp third_party/llama.cpp --out cv_llm_q4_0.gguf
"""
import argparse
import ctypes
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_LIB = os.path.join(HERE, "libcv_llm.so")


def install(model, gguf, lib=DEFAULT_LIB, prefill_threads=4, decode_threads=2, n_ctx=4096):
    """Swap the backbone of a loaded CosyVoice3 (cosyvoice.cli.cosyvoice.CosyVoice3)."""
    import numpy as np
    import torch

    shim = ctypes.CDLL(lib)
    shim.cv_llm_init.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int]
    shim.cv_llm_step.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]
    shim.cv_llm_set_threads.argtypes = [ctypes.c_int]
    n_embd = shim.cv_llm_init(gguf.encode(), n_ctx, decode_threads)
    if n_embd <= 0:
        raise RuntimeError(f"llama.cpp failed to load {gguf} ({n_embd})")
    hidden = np.empty(n_embd, dtype=np.float32)
    current = {"threads": decode_threads}

    def forward_one_step(xs, masks, cache=None):
        threads = prefill_threads if cache is None else decode_threads
        if current["threads"] != threads:
            shim.cv_llm_set_threads(threads)
            current["threads"] = threads
        if cache is None:
            shim.cv_llm_reset()
        x = np.ascontiguousarray(xs[0].detach().float().numpy())
        rc = shim.cv_llm_step(x.ctypes.data, x.shape[0], hidden.ctypes.data)
        if rc != 0:
            raise RuntimeError(f"llama.cpp decode failed ({rc})")
        return torch.from_numpy(hidden.copy()).to(xs.dtype).view(1, 1, -1), True

    backbone = model.model.llm.llm
    backbone.forward_one_step = forward_one_step
    # Only embed_tokens is still read on the torch side.
    backbone.model.model.layers = torch.nn.ModuleList()
    return shim


def export(model_dir, llama_cpp, out, quant="Q4_0"):
    """llm.pt's Qwen2 backbone -> f32 GGUF -> `quant` GGUF (quantized once, from f32)."""
    import torch
    from safetensors.torch import save_file

    sys.path.insert(0, os.path.join(llama_cpp, "gguf-py"))
    sys.path.insert(0, llama_cpp)
    import convert_hf_to_gguf
    from conversion import base

    # CosyVoice-BlankEN ships a modified tokenizer the converter cannot fingerprint; it is never
    # used (CosyVoice feeds embeddings), so label it.
    base.TextModel.get_vocab_base_pre = lambda self, tokenizer: "qwen2"
    state = torch.load(os.path.join(model_dir, "llm.pt"), map_location="cpu", weights_only=True, mmap=True)
    with tempfile.TemporaryDirectory() as tmp:
        # the converter names the model after this directory; keep it fixed so exports are
        # byte-reproducible
        hf = os.path.join(tmp, "CosyVoice3-LM-backbone")
        os.mkdir(hf)
        qwen_dir = os.path.join(model_dir, "CosyVoice-BlankEN")
        for name in os.listdir(qwen_dir):
            if name.endswith((".json", ".txt")):
                shutil.copy(os.path.join(qwen_dir, name), hf)
        # lm_head is tied to embed_tokens (and unused: CosyVoice has its own head)
        save_file({k[len("llm.model."):]: v.float().contiguous() for k, v in state.items()
                   if k.startswith("llm.model.") and k != "llm.model.lm_head.weight"},
                  os.path.join(hf, "model.safetensors"), metadata={"format": "pt"})
        f32 = os.path.join(hf, "backbone-f32.gguf")
        sys.argv = ["convert_hf_to_gguf.py", hf, "--outtype", "f32", "--outfile", f32]
        convert_hf_to_gguf.main()
        quantize = os.path.join(llama_cpp, "build", "bin", "llama-quantize")
        subprocess.run([quantize, f32, out + ".tmp", quant], check=True)
    os.replace(out + ".tmp", out)


def main():
    ap = argparse.ArgumentParser(description="Export CosyVoice3's LM backbone to a GGUF for llama.cpp.")
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--llama-cpp", required=True, help="llama.cpp checkout built by `make llm`")
    ap.add_argument("--out", required=True)
    ap.add_argument("--quant", default="Q4_0")
    args = ap.parse_args()
    export(args.model_dir, args.llama_cpp, args.out, args.quant)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
