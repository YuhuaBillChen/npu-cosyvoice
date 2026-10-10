"""Start whisper-xdna's STT server with a cap on its NPU hardware contexts and better decoding.

XDNA1 has 6 contexts; the TTS holds one, so whisper-xdna may use at most 5 (NPU_STT_MAX_CTX).
Its server has no setting for that, so this sets rawxrt.RawRT's default before the server runs.
Measured on whisper-xdna's own serving path: a 5-context ceiling costs no speed.

Its decoder also has no language detection or prompt: a request without a language was decoded
in NPU_STT_LANG, and Chinese came back without punctuation and in mixed simplified/traditional
script. With NPU_STT_LANG=auto the language is detected from the encoder output, and Chinese is
decoded with a short punctuated simplified-Chinese prompt (NPU_STT_PROMPT_ZH), which Whisper
imitates; other languages decode as before.
"""
import dataclasses
import os
import runpy
import sys

ROOT = "/opt/npu-whisper"
sys.path.insert(0, f"{ROOT}/src")
import rawxrt  # noqa: E402

rawxrt.RawRT.__init__.__defaults__ = (int(os.environ.get("NPU_STT_MAX_CTX", "5")),)

import whisper  # noqa: E402  (the server imports this same module)

PROMPT_ZH = os.environ.get("NPU_STT_PROMPT_ZH", "以下是普通话的句子。")
_decode = whisper.decode


def decode(model, mel, options=whisper.DecodingOptions(), **kw):
    if options.language in (None, "", "auto"):
        _, probs = model.detect_language(mel)
        probs = probs[0] if isinstance(probs, list) else probs
        options = dataclasses.replace(options, language=max(probs, key=probs.get))
    if options.language == "zh" and PROMPT_ZH and not options.prompt:
        options = dataclasses.replace(options, prompt=PROMPT_ZH)
    return _decode(model, mel, options, **kw)


whisper.decode = decode
os.chdir(ROOT)
sys.argv = [f"{ROOT}/serve/npu_stt_server.py"]
runpy.run_path(sys.argv[0], run_name="__main__")
