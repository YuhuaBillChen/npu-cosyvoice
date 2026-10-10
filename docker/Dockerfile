# Build/runtime base: drakosha/whisper-xdna's image already carries the XDNA userspace (XRT 2.21.75
# from trixie-backports), MLIR-AIE/IRON (with the Peano compiler). This adds a C++ toolchain
# (cmake for llama.cpp, if the LM runs in the same container) and the matching libxrt-dev headers,
# which backports no longer serves at 2.21.75.
# pinned to the image the published numbers were measured with (2026-07-26 build)
FROM ghcr.io/drakosha/npu-whisper@sha256:04291377e5266717ecb9e5c889378e93d798d5b9a7b8ce2a41ba7116eba052a4 AS base

RUN apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends g++ make cmake git uuid-dev \
    && rm -rf /var/lib/apt/lists/*
# libxrt-dev 1:2.21.75+dfsg-9~bpo13+1 from snapshot.debian.org (addressed by sha1), headers only.
ADD https://snapshot.debian.org/file/079dc64ae39a82df62a456dd2960349e70da0992 /tmp/libxrt-dev.deb
RUN echo "e51b375ac4ce8bac960bd6fdc883c22444c3e9ce092766995fcec4d4eff80b4c  /tmp/libxrt-dev.deb" | sha256sum -c - \
    && dpkg-deb -x /tmp/libxrt-dev.deb /opt/xrt-dev && rm /tmp/libxrt-dev.deb

WORKDIR /opt/npu-cosyvoice

# The voice bundle (voice/compose.yml): CosyVoice and its Python environment, this repository and its
# libraries built in. Python 3.11 because CosyVoice's pinned numpy 1.26 has no 3.13 wheels.
FROM base AS voice
COPY --from=ghcr.io/astral-sh/uv:0.11.32 /uv /usr/local/bin/uv
ENV UV_PYTHON_INSTALL_DIR=/opt/python UV_NO_CACHE=1
COPY voice/requirements.txt /tmp/requirements.txt
RUN uv venv --python 3.11 /opt/venv \
    && uv pip install --python /opt/venv/bin/python --index-url https://download.pytorch.org/whl/cpu \
        torch==2.14.0 torchaudio==2.11.0 \
    && uv pip install --python /opt/venv/bin/python -r /tmp/requirements.txt
# CosyVoice at the commit the published numbers were measured with
RUN git clone --quiet https://github.com/FunAudioLLM/CosyVoice.git /opt/CosyVoice \
    && git -C /opt/CosyVoice checkout --quiet 074ca6dc9e80a2f424f1f74b48bdd7d3fea531cc \
    && git -C /opt/CosyVoice submodule update --quiet --init third_party/Matcha-TTS \
    && rm -rf /opt/CosyVoice/.git
COPY . /opt/npu-cosyvoice
RUN make all
ENV PYTHONPATH=/opt/CosyVoice:/opt/CosyVoice/third_party/Matcha-TTS:/opt/npu-cosyvoice PATH=/opt/venv/bin:$PATH
CMD ["/opt/venv/bin/python", "voice/server.py"]

# whisper-xdna's STT server (published image, unmodified) with voice/whisper_entry.py, which caps its
# NPU contexts at 5 and adds language detection and the Chinese prompt
FROM ghcr.io/drakosha/npu-whisper@sha256:04291377e5266717ecb9e5c889378e93d798d5b9a7b8ce2a41ba7116eba052a4 AS whisper
COPY voice/whisper_entry.py /voice/whisper_entry.py
ENV PYTHONPATH=/usr/lib/python3/dist-packages \
    PEANO_INSTALL_DIR=/opt/mlir-aie/ironenv/lib/python3.13/site-packages/llvm-aie
CMD ["/opt/npu-whisper/refenv/bin/python", "-u", "/voice/whisper_entry.py"]

# the library's own image (the default target): toolchain only, the checkout is mounted
FROM base
