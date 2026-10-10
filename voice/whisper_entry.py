"""Start whisper-xdna's STT server with a cap on its NPU hardware contexts and better decoding.

XDNA1 has 6 contexts; the TTS holds one, so whisper-xdna may use at most 5 (NPU_STT_MAX_CTX).
Its server has no setting for that, so this sets rawxrt.RawRT's default before the server runs.
Measured on whisper-xdna's own serving path: a 5-context ceiling costs no speed.

Its decoder also has no language detection or prompt: a request without a language was decoded
in NPU_STT_LANG, and Chinese came back without punctuation and in mixed simplified/traditional
script. With NPU_STT_LANG=auto the language is detected from the encoder output, and Chinese is
decoded with a short punctuated simplified-Chinese prompt (NPU_STT_PROMPT_ZH), which Whisper
imitates (with ASCII marks after Chinese characters made full-width); other languages decode
as before.

It also drops the encoder matrices the server keeps but never reads (~2.5 GB).
"""
import dataclasses
import os
import re
import runpy
import sys

ROOT = "/opt/npu-whisper"
sys.path.insert(0, f"{ROOT}/src")
import rawxrt  # noqa: E402

rawxrt.RawRT.__init__.__defaults__ = (int(os.environ.get("NPU_STT_MAX_CTX", "5")),)

import whisper  # noqa: E402  (the server imports this same module)

PROMPT_ZH = os.environ.get("NPU_STT_PROMPT_ZH", "以下是普通话的句子。")
# the prompt makes Chinese punctuated, but often with ASCII marks ("你好,今天"); after a Chinese
# character they become the full-width ones
ZH_PUNCT = {",": "，", ".": "。", "?": "？", "!": "！", ":": "：", ";": "；"}
FULLWIDTH = re.compile(r"([\u4e00-\u9fff])([,.?!:;])")
_decode = whisper.decode


def decode(model, mel, options=whisper.DecodingOptions(), **kw):
    if options.language in (None, "", "auto"):
        _, probs = model.detect_language(mel)
        probs = probs[0] if isinstance(probs, list) else probs
        options = dataclasses.replace(options, language=max(probs, key=probs.get))
    if options.language == "zh" and PROMPT_ZH and not options.prompt:
        options = dataclasses.replace(options, prompt=PROMPT_ZH)
    result = _decode(model, mel, options, **kw)
    if options.language != "zh":
        return result
    fix = lambda r: dataclasses.replace(r, text=FULLWIDTH.sub(lambda m: m.group(1) + ZH_PUNCT[m.group(2)], r.text))
    return [fix(r) for r in result] if isinstance(result, list) else fix(result)


whisper.decode = decode


import npu_whisper_encoder  # noqa: E402  (the server imports this module inside its loader)

_prepare = npu_whisper_encoder.prepare_mlp_chunks


def prepare_mlp_chunks(W, *a, **kw):
    # the server keeps fp32 copies of every encoder matrix both as loaded and transposed, but the
    # encoder only reads the transposed ones; dropping the others saves ~2.5 GB of RAM
    for k in [k for k in W if k.endswith(".weight") and k[:-len("weight")] + "weight_T" in W]:
        del W[k]
    return _prepare(W, *a, **kw)


npu_whisper_encoder.prepare_mlp_chunks = prepare_mlp_chunks

os.chdir(ROOT)
sys.argv = [f"{ROOT}/serve/npu_stt_server.py"]
runpy.run_path(sys.argv[0], run_name="__main__")
