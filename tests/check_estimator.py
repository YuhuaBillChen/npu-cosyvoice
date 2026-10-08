"""Check the NPU estimator against torch on a real estimator input, then check the priority gate.

    python tests/check_estimator.py --model-dir pretrained_models/Fun-CosyVoice3-0.5B-2512 \
        --weights weights --aot aot --prompt-wav asset/zero_shot_prompt.wav

Prints the relative error of the NPU output against the fp32 torch estimator (int8 W8A8 with
SmoothQuant q/k/v is ~3.3% on the estimator output), the median forward time, and the time a forward waited while
another thread held the gate for 0.5 s (output must be unchanged).
"""
import argparse
import json
import os
import threading
import time

import torch

import npu_cosyvoice
from npu_cosyvoice.export import PROMPT_TEXT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--aot", required=True)
    ap.add_argument("--prompt-wav", required=True)
    ap.add_argument("--gate", default="/dev/shm/npu-cosyvoice-test.gate")
    args = ap.parse_args()

    from cosyvoice.cli.cosyvoice import AutoModel
    npu_cosyvoice.soundfile_audio()
    model = AutoModel(model_dir=args.model_dir)
    est = model.model.flow.decoder.estimator
    torch_forward = est.forward
    captured = {}

    def capture(*a, **k):
        captured.setdefault("args", (a, k))
        return torch_forward(*a, **k)

    est.forward = capture
    list(model.inference_zero_shot("Today's meeting is about the plan for next quarter.", PROMPT_TEXT, args.prompt_wav, stream=False))
    a, k = captured["args"]
    with torch.no_grad():
        y_ref = torch_forward(*a, **k).float()

    ne = npu_cosyvoice.install(model, args.weights, args.aot, gate=args.gate)
    ts = []
    for _ in range(5):
        t0 = time.perf_counter()
        y = est.forward(*a, **k).float()
        ts.append(time.perf_counter() - t0)
    rel = float((y - y_ref).norm() / y_ref.norm())

    gate = npu_cosyvoice.NpuGate(args.gate)
    holding = threading.Event()

    def holder():
        with gate.hold():
            holding.set()
            time.sleep(0.5)

    th = threading.Thread(target=holder)
    th.start()
    holding.wait()
    p0 = ne.paused_seconds()
    t0 = time.perf_counter()
    y_gated = est.forward(*a, **k).float()
    gated_s = time.perf_counter() - t0
    th.join()
    print(json.dumps(dict(frames=int(a[0].shape[-1]), rel_err_vs_torch=round(rel, 4), npu_ms=round(sorted(ts)[2] * 1e3, 1),
                          gated_forward_ms=round(gated_s * 1e3, 1), paused_ms=round((ne.paused_seconds() - p0) * 1e3, 1),
                          gated_output_identical=bool(torch.equal(y, y_gated)), errors=ne.errors)))
    os.remove(args.gate)


if __name__ == "__main__":
    main()
