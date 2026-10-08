"""Start whisper-xdna's STT server with a cap on its NPU hardware contexts.

XDNA1 has 6 contexts; the TTS holds one, so whisper-xdna may use at most 5 (NPU_STT_MAX_CTX).
Its server has no setting for that, so this sets rawxrt.RawRT's default before the server runs.
Measured on whisper-xdna's own serving path: a 5-context ceiling costs no speed.
"""
import os
import runpy
import sys

ROOT = "/opt/npu-whisper"
sys.path.insert(0, f"{ROOT}/src")
import rawxrt  # noqa: E402

rawxrt.RawRT.__init__.__defaults__ = (int(os.environ.get("NPU_STT_MAX_CTX", "5")),)
os.chdir(ROOT)
sys.argv = [f"{ROOT}/serve/npu_stt_server.py"]
runpy.run_path(sys.argv[0], run_name="__main__")
