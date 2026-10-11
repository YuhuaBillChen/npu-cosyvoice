#!/bin/bash
# One-time preparation of $DATA_DIR for voice/server.py; each step is skipped once its output exists.
#   1. CosyVoice3 checkpoint (ModelScope, ~3 GB)
#   2. int8 DiT weights (calibrates on a few sentences, a few minutes on the CPU)
#   3. the LM backbone as a Q4_0 GGUF
#   4. the NPU overlay for $NPU_DEVICE (npu1 = XDNA1, npu2 = XDNA2, untested)
# Steps 2-4 are deterministic, so by default they are downloaded prebuilt from $PREBUILT_REPO on
# Hugging Face (checksummed) when the checkpoint matches the tested one; PREBUILT_REPO= builds them here.
set -euo pipefail
DATA="${DATA_DIR:-/data}"
MODEL="$DATA/Fun-CosyVoice3-0.5B-2512"
PY=/opt/venv/bin/python
cd /opt/npu-cosyvoice

if [ ! -f "$MODEL/hift.pt" ] || [ ! -f "$MODEL/llm.pt" ] || [ ! -f "$MODEL/flow.pt" ]; then
  echo "== downloading CosyVoice3"  # resumes an interrupted download
  # minus what serving never loads: the RL-tuned LM, the TensorRT estimator, the batch tokenizer
  $PY -c "from modelscope import snapshot_download; snapshot_download('FunAudioLLM/Fun-CosyVoice3-0.5B-2512', local_dir='$MODEL',
      ignore_patterns=['llm.rl.pt', 'flow.decoder.estimator.fp32.onnx', 'speech_tokenizer_v3.batch.onnx'])"
fi
# the checkpoint the published numbers were measured with (ModelScope has no pinned revision)
TESTED=1
for f in "69f43bd545131c30 llm.pt" "a6fab32a7825e5b0 flow.pt" "b279d7641eb97ae5 hift.pt"; do
  set -- $f
  if [ "$(sha256sum "$MODEL/$2" | cut -c1-16)" != "$1" ]; then
    echo "note: $2 differs from the checkpoint this was tested with; building the int8 files here"
    TESTED=0
  fi
done
PREBUILT_REPO="${PREBUILT_REPO-billchen42/npu-cosyvoice}"
if [ -n "$PREBUILT_REPO" ] && [ "$TESTED" = 1 ] && [ ! -f "$DATA/weights/meta.json" ]; then
  echo "== downloading prebuilt int8 weights, LM and NPU overlay from $PREBUILT_REPO"
  PRE="$DATA/prebuilt"
  # a failed download (offline, repository missing) is not an error: the files are built below
  $PY -c "from huggingface_hub import snapshot_download; snapshot_download('$PREBUILT_REPO', local_dir='$PRE',
      allow_patterns=['weights/*', 'cv_llm_q4_0.gguf', 'aot/${NPU_DEVICE:-npu1}/*', 'SHA256SUMS'])" 2>"$PRE.err" \
    && (cd "$PRE" && grep -E " \./(weights/|cv_llm_q4_0\.gguf|aot/${NPU_DEVICE:-npu1}/)" SHA256SUMS | sha256sum --quiet -c -) \
    && mv "$PRE/weights" "$PRE/cv_llm_q4_0.gguf" "$DATA/" \
    && { [ ! -d "$PRE/aot/${NPU_DEVICE:-npu1}" ] || { mkdir -p "$DATA/aot" && mv "$PRE/aot/${NPU_DEVICE:-npu1}"/* "$DATA/aot/"; }; } \
    || echo "note: prebuilt download failed ($(tail -n 1 "$PRE.err" 2>/dev/null | cut -c1-120)); building the int8 files here"
  rm -f "$PRE.err"
  rm -rf "$PRE"
fi
if [ ! -f "$DATA/weights/meta.json" ]; then
  echo "== exporting int8 DiT weights"
  $PY -m npu_cosyvoice.export --model-dir "$MODEL" --prompt-wav /opt/CosyVoice/asset/zero_shot_prompt.wav --out "$DATA/weights"
fi
if [ ! -f "$DATA/cv_llm_q4_0.gguf" ]; then
  echo "== exporting the LM backbone"
  $PY -m npu_cosyvoice.llm --model-dir "$MODEL" --llama-cpp third_party/llama.cpp --out "$DATA/cv_llm_q4_0.gguf"
fi
if [ ! -f "$DATA/aot/mmi8i32_512x1024x1024_64_64_64.xclbin" ]; then
  echo "== compiling the NPU overlay (${NPU_DEVICE:-npu1})"
  python3 scripts/compile_xclbin.py --device "${NPU_DEVICE:-npu1}" --out "$DATA/aot"
fi
mkdir -p "$DATA/voices"
echo "== ready: $DATA"
