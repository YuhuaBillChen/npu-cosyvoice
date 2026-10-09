# npu-cosyvoice

CosyVoice3's flow-matching DiT on the AMD XDNA1 NPU (Ryzen AI 7040/8040, "Phoenix"/"Hawk Point").

CosyVoice3 spends most of its time in the flow decoder: a 22-layer DiT, run with classifier-free
guidance (batch 2) for 10 Euler steps per sentence. On a laptop CPU that alone is slower than real
time, and it competes for cores with everything else on the machine. This library moves every GEMM
of the DiT to the NPU in int8 and keeps the rest (norms, RoPE, attention, GELU) in AVX-512 C++ on a
few host threads.

It also runs CosyVoice's LM backbone (Qwen2 0.5B) through llama.cpp on the CPU, which is the other
half of the time once the flow is off the CPU.

Technical report: [Let the Idle NPU Speak: Local Voice Chat on a Ryzen AI Laptop](docs/npu-cosyvoice-report.pdf).
Demo videos and audio samples: <https://yuhuabillchen.github.io/npu-cosyvoice/>.

Status: runs in production on one machine (voice chat next to a 27B LLM on the same SoC, sharing
the NPU with a Whisper STT service).

## Quick start

Any XDNA1 machine (Ryzen AI 7040/8040) running Linux 6.14+ with Docker:

```sh
git clone https://github.com/YuhuaBillChen/npu-cosyvoice.git && cd npu-cosyvoice
export RENDER_GID=$(getent group render | cut -d: -f3)
docker compose -f voice/compose.yml run --rm setup        # once: model download, int8 export, NPU overlay (~15 min)
docker compose -f voice/compose.yml --profile stt up -d   # TTS + STT behind the OpenAI audio API on :8000
curl -s localhost:8000/v1/audio/speech -H 'Content-Type: application/json' \
    -d '{"input": "Hello from the NPU.", "response_format": "mp3"}' -o hello.mp3
```

That is the [voice bundle](voice/README.md): CosyVoice3 TTS with the flow DiT on the NPU, plus
whisper-xdna STT, usable from Open WebUI or any OpenAI-compatible client. The rest of this README
is the library underneath it.

## What runs where

| part | where |
|---|---|
| LM backbone (Qwen2 0.5B) | CPU, llama.cpp Q4_0 (`npu_cosyvoice.llm`); 4 threads for the prompt, 2 per token |
| LM embeddings, speech-token head, sampling | unchanged (torch) |
| flow DiT, 22 blocks: q/k/v, out, FF GEMMs | **NPU**, int8 x int8 -> int32, one 512x1024x1024 overlay, one hw context |
| flow DiT: LayerNorm + AdaLN, RoPE, attention (bf16 dot products), GELU, (de)quantization | host C++, AVX-512, 4 threads |
| time/input embedding, output projection, HiFT vocoder | unchanged (torch) |

Weights are W8A8: per-output-channel int8 weights, per-token int8 activations, with SmoothQuant
(alpha 0.5; Xiao et al., ICML 2023, arXiv:2211.10438) on the q/k/v input, the layer most sensitive
to int8.

## Numbers (Ryzen 7 8845HS)

| | |
|---|---|
| one DiT block, 956 frames, CFG batch 2 | CPU bf16 (8 threads) 20 ms -> NPU (6 host threads) 7.4 ms; 4 host threads 8.3 ms |
| one estimator call (22 blocks), 318 frames | 163 ms; output within 3.3% of the fp32 torch estimator |
| mel error vs bf16 at 10 steps (lower is better) | NPU int8, 10 steps: 3.0% (int8 rounding only); bf16 cut to 5 steps to save CPU time: 7.4% (error from the steps, not bf16) |

Real-time factor (synthesis time / audio length, lower is better), 10 flow steps:

| | torch LM | llama.cpp LM |
|---|---|---|
| `examples/synthesize.py`: one sentence, warm, upstream fp32 loading, idle machine | 2.7 | **0.83** |
| a deployment (bf16 weights, 3 sentences, serial, 4 threads, Q8_0 LM): CPU flow -> NPU flow | | 1.44 -> **0.84** |
| same, while a 27B LLM generates on the iGPU | | 2.13 -> **1.33** |
| NPU flow with the LM of sentence N+1 overlapping the flow of N (4 sentences, Q4_0 LM, 2+4 threads): idle / 27B generating | | **0.64** / **1.02** |

## Pieces

- `csrc/dit_npu.cpp` -> `npu_cosyvoice/libdit_npu.so`: the DiT. Errors are returned, never abort.
- `npu_cosyvoice.install(model, weights_dir, aot_dir)`: swaps the estimator of a loaded CosyVoice3.
- `npu_cosyvoice.AdaptiveSteps`: for sentence-by-sentence playback, picks per sentence the most
  flow steps (4..10) that still finish before the previous sentence stops playing, and the fewest
  for the first sentence of a reply.
- `npu_cosyvoice.NpuGate`: the NPU has a 6-context ceiling and no preemption. Another NPU user
  (e.g. whisper-xdna STT) holds the gate while it runs; the DiT pauses at the next layer
  boundary. Measured: STT next to a speaking TTS is within 3% of STT alone.
- `npu_cosyvoice.llm.install(model, gguf)`: swaps the LM backbone for llama.cpp
  (`csrc/cv_llm.c` -> `npu_cosyvoice/libcv_llm.so`); `python -m npu_cosyvoice.llm` exports the GGUF.
- `npu_cosyvoice/export.py`: calibration and int8 export.
- `scripts/compile_xclbin.py`: compiles the one NPU overlay.
- `voice/`: a docker compose bundle serving this TTS and whisper-xdna STT behind the OpenAI audio
  API, so any chat client gets a voice; see [voice/README.md](voice/README.md).

## Build and run

Needs the amdxdna kernel driver (mainline since Linux 6.14) and Docker. The NPU device belongs to
the `render` group, hence `--group-add`.

```sh
docker build -t npu-cosyvoice -f docker/Dockerfile .
# NPU DiT library, llama.cpp (pinned commit) and the LM shim
docker run --rm --user $(id -u):$(id -g) -v $PWD:/opt/npu-cosyvoice npu-cosyvoice make all
# the NPU overlay (add --device npu2 for XDNA2, see Notes)
docker run --rm --user $(id -u):$(id -g) -e HOME=/tmp -v $PWD:/opt/npu-cosyvoice \
    -e NPU_AOT_DIR=/opt/npu-cosyvoice/aot -e PYTHONPATH=/usr/lib/python3/dist-packages \
    npu-cosyvoice python3 scripts/compile_xclbin.py
# the overlay on this NPU, bit-exact against numpy (needs the NPU passed through, see below)
docker run --rm ... -e PYTHONPATH=/usr/lib/python3/dist-packages npu-cosyvoice python3 tests/check_gemm.py --aot aot
```

Then the export, the example and the checks. The voice bundle's image has CosyVoice and the NPU
passthrough ready (after `docker compose -f voice/compose.yml run --rm setup`, the model is in
`/data`):

```sh
docker compose -f voice/compose.yml run --rm voice bash
M=/data/Fun-CosyVoice3-0.5B-2512; W=/opt/CosyVoice/asset/zero_shot_prompt.wav
python -m npu_cosyvoice.export --model-dir $M --prompt-wav $W --out weights
python -m npu_cosyvoice.llm --model-dir $M --llama-cpp third_party/llama.cpp --out cv_llm_q4_0.gguf
python examples/synthesize.py --model-dir $M --weights weights --aot /data/aot --prompt-wav $W \
    --llm-gguf cv_llm_q4_0.gguf --repeat 3 "Hello, this sentence was synthesized on the NPU."
python tests/check_estimator.py --model-dir $M --weights weights --aot /data/aot --prompt-wav $W
```

Outside that image you need CosyVoice's source and `third_party/Matcha-TTS` on `PYTHONPATH`, its
requirements (`voice/requirements.txt`), and the NPU passed through (`--device /dev/accel/accel0
--group-add $(getent group render | cut -d: -f3) --ulimit memlock=-1 --cap-add IPC_LOCK`).

The GGUF export is byte-reproducible for a given llama.cpp commit.

## Notes

- If the NPU runtime-suspends while idle, resuming it allocates contiguous kernel memory, which
  can fail under memory pressure (`aie2_hw_start: Allocate async events failed, ret -12`). Keeping
  it awake avoids that:
  `SUBSYSTEM=="pci", ACTION=="add|bind", DRIVER=="amdxdna", ATTR{power/control}="on"` in udev.
- XDNA1 has 6 hardware contexts per device and this library holds one; whatever else uses the NPU
  has to leave one free.
- The host code is built with `-march=znver4` (AVX-512, bf16 dot products), which every XDNA1 and
  XDNA2 SoC has.
- Tested on XDNA1 only (Ryzen AI 7040/8040). XDNA2 (Ryzen AI 300, Strix Halo, Krackan) should only
  need the overlay rebuilt with `scripts/compile_xclbin.py --device npu2` (8 columns instead of 4);
  it compiles, but we have no XDNA2 machine to run it. If you do, `tests/check_gemm.py` and
  `tests/check_estimator.py` results are very welcome in an issue.
- Streaming (CosyVoice's `stream=True`) works (`dit_forward_chunked` implements the block-causal
  mask), but sounded worse in our listening tests than sentence-by-sentence synthesis.

## License

Apache-2.0. See NOTICE for the projects this builds on.
