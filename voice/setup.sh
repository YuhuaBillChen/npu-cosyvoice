#!/bin/bash
# One-time preparation of $DATA_DIR for voice/server.py; each step is skipped once its output exists.
#   1. CosyVoice3 checkpoint (ModelScope, ~3 GB)
#   2. int8 DiT weights (calibrates on a few sentences, a few minutes on the CPU)
#   3. the LM backbone as a Q4_0 GGUF
#   4. the NPU overlay for $NPU_DEVICE (npu1 = XDNA1, npu2 = XDNA2, untested)
# Steps 2-4 are deterministic, so by default they are downloaded prebuilt from $PREBUILT_REPO on
# Hugging Face (checksummed); PREBUILT_REPO= builds them here.
set -euo pipefail
DATA="${DATA_DIR:-/data}"
MODEL="$DATA/Fun-CosyVoice3-0.5B-2512"
PY=/opt/venv/bin/python
cd /opt/npu-cosyvoice

# Every checkpoint file the server loads, pinned by hash in voice/cosyvoice3.sha256 (ModelScope has
# no pinned revision): a changed or tampered file stops setup instead of being loaded.
if ! $PY voice/checksums.py voice/cosyvoice3.sha256 "$MODEL" CosyVoice-BlankEN 2>/dev/null; then
  echo "== downloading CosyVoice3"  # resumes an interrupted download
  $PY -c "from modelscope import snapshot_download; snapshot_download('FunAudioLLM/Fun-CosyVoice3-0.5B-2512', local_dir='$MODEL',
      allow_patterns=[l.split()[1] for l in open('voice/cosyvoice3.sha256')])"
  if ! $PY voice/checksums.py voice/cosyvoice3.sha256 "$MODEL" CosyVoice-BlankEN; then
    echo "error: the CosyVoice3 download differs from the checkpoint this release was tested with; stopping" >&2
    exit 1
  fi
fi
# The prebuilt files are checked against SHA256SUMS at a pinned revision, and that list against the
# hash below, so neither the repository's current state nor a mirror (PREBUILT_REPO) is trusted.
PREBUILT_REPO="${PREBUILT_REPO-billchen42/npu-cosyvoice}"
PREBUILT_REVISION=522e671dcfa993eb3d60c2ffeb3b96b176f791a9
PREBUILT_SUMS=82d2175c6b8aa5451c3a9eb01815fb7621ee009321329e5626e0d0acc3404b3d
DEV="${NPU_DEVICE:-npu1}"
if [ -n "$PREBUILT_REPO" ] && [ ! -f "$DATA/weights/meta.json" ]; then
  echo "== downloading prebuilt int8 weights, LM and NPU overlay from $PREBUILT_REPO"
  PRE="$DATA/prebuilt"
  rm -rf "$PRE"
  # a failed download or check is not an error: the files are built below from the checked checkpoint
  $PY -c "from huggingface_hub import snapshot_download; snapshot_download('$PREBUILT_REPO', local_dir='$PRE',
      revision='$PREBUILT_REVISION' if '$PREBUILT_REPO' == 'billchen42/npu-cosyvoice' else None,
      allow_patterns=['weights/*', 'cv_llm_q4_0.gguf', 'aot/$DEV/*', 'SHA256SUMS'])" 2>"$PRE.err" \
    && echo "$PREBUILT_SUMS  $PRE/SHA256SUMS" | sha256sum --quiet -c - 2>>"$PRE.err" \
    && grep -E " \./(weights/|cv_llm_q4_0\.gguf|aot/$DEV/)" "$PRE/SHA256SUMS" >"$PRE.sums" \
    && $PY voice/checksums.py "$PRE.sums" "$PRE" weights "aot/$DEV" 2>>"$PRE.err" \
    && mv "$PRE/weights" "$PRE/cv_llm_q4_0.gguf" "$DATA/" \
    && { [ ! -d "$PRE/aot/$DEV" ] || { mkdir -p "$DATA/aot" && mv "$PRE/aot/$DEV"/* "$DATA/aot/"; }; } \
    || echo "note: prebuilt download failed ($(tail -n 1 "$PRE.err" 2>/dev/null | cut -c1-120)); building the int8 files here"
  rm -rf "$PRE" "$PRE.err" "$PRE.sums"
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
