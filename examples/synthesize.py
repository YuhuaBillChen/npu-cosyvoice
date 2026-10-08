"""Synthesize one sentence with the DiT on the NPU and report the real-time factor.

    python examples/synthesize.py --model-dir pretrained_models/Fun-CosyVoice3-0.5B-2512 \
        --weights weights --aot aot --prompt-wav asset/zero_shot_prompt.wav --out out.wav \
        --llm-gguf cv_llm_q4_0.gguf "The flow decoder now runs on the NPU, so the CPU is free for everything else."
"""
import argparse
import time

import soundfile
import torch

import npu_cosyvoice
import npu_cosyvoice.llm
from npu_cosyvoice.export import PROMPT_TEXT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("text")
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--aot", required=True)
    ap.add_argument("--prompt-wav", required=True)
    ap.add_argument("--prompt-text", default=PROMPT_TEXT)
    ap.add_argument("--steps", type=int, default=10, help="Euler steps of the flow (CosyVoice3 default 10)")
    ap.add_argument("--threads", type=int, default=4, help="host threads of the NPU library")
    ap.add_argument("--llm-gguf", help="run the LM backbone through llama.cpp (see npu_cosyvoice/llm.py)")
    ap.add_argument("--torch-threads", type=int, default=4, help="for the parts left in torch (embeddings, vocoder)")
    ap.add_argument("--repeat", type=int, default=1, help="synthesize this many times; the first run includes warm-up")
    ap.add_argument("--out", default="out.wav")
    args = ap.parse_args()

    from cosyvoice.cli.cosyvoice import AutoModel
    torch.set_num_threads(args.torch_threads)
    npu_cosyvoice.soundfile_audio()
    model = AutoModel(model_dir=args.model_dir)
    npu_cosyvoice.install(model, args.weights, args.aot, threads=args.threads)
    npu_cosyvoice.AdaptiveSteps(min_steps=args.steps, max_steps=args.steps).install(model)
    if args.llm_gguf:
        npu_cosyvoice.llm.install(model, args.llm_gguf)

    # encode the reference voice once instead of on every call
    model.add_zero_shot_spk(args.prompt_text, args.prompt_wav, "voice")
    for i in range(args.repeat):
        t0 = time.perf_counter()
        out = model.inference_zero_shot(args.text, "", "", zero_shot_spk_id="voice", stream=False)
        audio = torch.cat([o["tts_speech"] for o in out], 1)
        el = time.perf_counter() - t0
        dur = audio.shape[1] / model.sample_rate
        print(f"run {i + 1}: {dur:.2f} s audio in {el:.2f} s (RTF {el / dur:.2f})", flush=True)
    soundfile.write(args.out, audio[0].numpy(), model.sample_rate)
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
