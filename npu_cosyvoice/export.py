"""Export the CosyVoice3 DiT for libdit_npu: W8A8 int8 with SmoothQuant on q/k/v.

    python -m npu_cosyvoice.export --model-dir pretrained_models/Fun-CosyVoice3-0.5B-2512 \
        --prompt-wav asset/zero_shot_prompt.wav --out weights/

Per block it writes 1024x1024 int8 weight tiles ({qkv,o,f1,f2}_w_<k>_<n>.bin, K-major),
per-output-channel scales (*_sw.bin), biases (*_b.bin) and the inverse SmoothQuant scale for the
q/k/v input (qkv_inv_smooth.bin); plus the AdaLayerNormZero weights (adaln_*.npy) that the torch
side needs once the blocks are dropped. q/k/v are the int8-sensitive layers: smoothing them
(alpha 0.5, activation maxima from a few calibration sentences) roughly halves their error.
SmoothQuant: G. Xiao et al., "SmoothQuant: Accurate and Efficient Post-Training Quantization for
Large Language Models", ICML 2023, arXiv:2211.10438.
"""
import argparse
import json
import os

import numpy as np
import torch

TILE = 1024
DIM = 1024
# CosyVoice3 speaks Chinese and English; the activation ranges are calibrated on both.
CALIB = [
    "这家餐厅的牛肉面汤头很浓，面条也很有嚼劲，值得再来一次。",
    "请记得把报告的第二部分补充完整，周五之前发给我看一下。",
    "如果明天下雨的话，我们就把户外活动改到室内体育馆进行。",
    "The quick brown fox jumps over the lazy dog near the river bank.",
]
# transcript of CosyVoice's own reference recording, asset/zero_shot_prompt.wav (a Mandarin speaker)
PROMPT_TEXT = "You are a helpful assistant.<|endofprompt|>希望你以后能够做的比我还好呦。"


def _save(path, a, dtype=np.float32):
    np.ascontiguousarray(np.asarray(a, dtype=dtype)).tofile(path)


def _gemm(d, name, w, bias, smooth=None):
    wt = w.float().T.contiguous()  # K x N
    if smooth is not None:
        wt = wt * smooth[:, None]
    sw = wt.abs().amax(0).clamp(min=1e-8) / 127
    wq = torch.round(wt / sw[None, :]).clamp(-127, 127).to(torch.int8)
    K, N = wq.shape
    for kb in range(K // TILE):
        for nb in range(N // TILE):
            _save(f"{d}/{name}_w_{kb}_{nb}.bin", wq[kb * TILE:(kb + 1) * TILE, nb * TILE:(nb + 1) * TILE].numpy(), np.int8)
    _save(f"{d}/{name}_sw.bin", sw.numpy())
    _save(f"{d}/{name}_b.bin", bias.float().numpy())


@torch.no_grad()
def export(model, out, prompt_wav, prompt_text=PROMPT_TEXT, calib=CALIB, alpha=0.5):
    """model: a loaded CosyVoice3 (torch estimator, before npu_cosyvoice.flow.install)."""
    est = model.model.flow.decoder.estimator
    amax = {}

    def hook(i):
        def f(mod, inp):
            a = inp[0].float().abs().reshape(-1, DIM).amax(0)
            amax[i] = torch.maximum(amax[i], a) if i in amax else a
        return f

    hooks = [b.attn.to_q.register_forward_pre_hook(hook(i)) for i, b in enumerate(est.transformer_blocks)]
    torch.manual_seed(1)
    for s in calib:
        list(model.inference_zero_shot(s, prompt_text, prompt_wav, stream=False))
    for h in hooks:
        h.remove()

    os.makedirs(out, exist_ok=True)
    for i, b in enumerate(est.transformer_blocks):
        d = f"{out}/L{i:02d}"
        os.makedirs(d, exist_ok=True)
        wqkv = torch.cat([b.attn.to_q.weight, b.attn.to_k.weight, b.attn.to_v.weight], 0).float()
        bqkv = torch.cat([b.attn.to_q.bias, b.attn.to_k.bias, b.attn.to_v.bias]).float()
        ax = amax[i].clamp(min=1e-5)
        aw = wqkv.abs().amax(0).clamp(min=1e-5)
        smooth = (ax ** alpha / aw ** (1 - alpha)).clamp(min=1e-3)
        _save(f"{d}/qkv_inv_smooth.bin", (1 / smooth).numpy())
        _gemm(d, "qkv", wqkv, bqkv, smooth)
        _gemm(d, "o", b.attn.to_out[0].weight, b.attn.to_out[0].bias)
        _gemm(d, "f1", b.ff.ff[0][0].weight, b.ff.ff[0][0].bias)
        _gemm(d, "f2", b.ff.ff[2].weight, b.ff.ff[2].bias)
    blocks = est.transformer_blocks
    np.save(f"{out}/adaln_w_f16.npy", np.stack([b.attn_norm.linear.weight.float().numpy().astype(np.float16) for b in blocks]))
    np.save(f"{out}/adaln_b_f32.npy", np.stack([b.attn_norm.linear.bias.float().numpy() for b in blocks]))
    with open(f"{out}/meta.json", "w") as f:
        json.dump(dict(layers=len(blocks), calib_sentences=list(calib), smooth_alpha=alpha), f, indent=1)
    return len(blocks)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--prompt-wav", required=True, help="reference voice used while calibrating")
    ap.add_argument("--prompt-text", default=PROMPT_TEXT)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    from cosyvoice.cli.cosyvoice import AutoModel
    from . import soundfile_audio
    soundfile_audio()
    model = AutoModel(model_dir=args.model_dir)
    n = export(model, args.out, args.prompt_wav, args.prompt_text)
    print(f"exported {n} layers to {args.out}")


if __name__ == "__main__":
    main()
