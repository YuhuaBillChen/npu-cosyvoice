"""OpenAI-compatible speech endpoints on one Ryzen AI NPU: CosyVoice3 TTS, and STT forwarded to
whisper-xdna.

    POST /v1/audio/speech           {"input": ..., "voice": ..., "response_format": "mp3", "speed": 1.0}
    POST /v1/audio/transcriptions   multipart, forwarded to $STT_URL (whisper-xdna) if set
    GET  /v1/audio/voices, /v1/models, /health

TTS: the flow DiT runs int8 on the NPU, its steps adapt to the playback deadline, and the LM
backbone runs in llama.cpp. LM and token2wav are two pipelined stages (the LM of sentence N+1
overlaps token2wav of sentence N), each first-come first-served so sentences finish in order.
STT: while a transcription is in flight this process holds the NPU gate, so the DiT pauses at the
next layer boundary and speech recognition gets the NPU to itself.

Voices: "default" is CosyVoice's own reference recording. For another one, put <name>.wav (a few
seconds of clean speech) and <name>.txt (its transcript) in $VOICES_DIR. Unknown voice names,
such as OpenAI's "alloy", get the default voice.
"""
import ctypes
import gc
import glob
import io
import json
import os
import subprocess
import threading
import time
import urllib.error
import urllib.request

import numpy as np
import soundfile
import torch
from fastapi import FastAPI, HTTPException, Request, Response
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

import npu_cosyvoice
import npu_cosyvoice.llm
from npu_cosyvoice.export import PROMPT_TEXT

DATA = os.environ.get("DATA_DIR", "/data")
MODEL_DIR = os.environ.get("MODEL_DIR", f"{DATA}/Fun-CosyVoice3-0.5B-2512")
VOICES_DIR = os.environ.get("VOICES_DIR", f"{DATA}/voices")
COSYVOICE_DIR = os.environ.get("COSYVOICE_DIR", "/opt/CosyVoice")
STT_URL = os.environ.get("STT_URL", "").rstrip("/")
GATE = os.environ.get("NPU_GATE", "/dev/shm/npu-cosyvoice.gate")
FLOW_THREADS = int(os.environ.get("FLOW_THREADS", "4"))
PORT = int(os.environ.get("PORT", "8000"))
# clients such as Open WebUI send no speed; this is used when a request has none
DEFAULT_SPEED = float(os.environ.get("DEFAULT_SPEED", "1.0"))
# the endpoint has no authentication; these keep one client from filling memory or the queue
STT_MAX_BYTES = 25 * 1024 * 1024  # OpenAI's limit for this endpoint
STT_MAX_PENDING = 4
STT_PENDING = 0
INSTRUCT = "You are a helpful assistant.<|endofprompt|>"
FORMATS = {"mp3": ("audio/mpeg", ["-f", "mp3", "-b:a", "128k"]),
           "opus": ("audio/ogg", ["-f", "ogg", "-c:a", "libopus", "-b:a", "48k"]),
           "aac": ("audio/aac", ["-f", "adts", "-c:a", "aac", "-b:a", "128k"]),
           "flac": ("audio/flac", None), "wav": ("audio/wav", None), "pcm": ("audio/pcm", None)}


class FifoLock:
    """Ticket lock: waiters acquire strictly in arrival order (threading.Lock is not fair)."""

    def __init__(self):
        self.cond = threading.Condition()
        self.next = self.serving = 0

    def __enter__(self):
        with self.cond:
            mine, self.next = self.next, self.next + 1
            while self.serving != mine:
                self.cond.wait()

    def __exit__(self, *exc):
        with self.cond:
            self.serving += 1
            self.cond.notify_all()


def load():
    from cosyvoice.cli.cosyvoice import AutoModel

    torch.set_num_threads(FLOW_THREADS)
    npu_cosyvoice.soundfile_audio()
    model = AutoModel(model_dir=MODEL_DIR)
    npu_cosyvoice.install(model, f"{DATA}/weights", f"{DATA}/aot", threads=FLOW_THREADS, gate=GATE)
    steps = npu_cosyvoice.AdaptiveSteps().install(model)
    npu_cosyvoice.llm.install(model, f"{DATA}/cv_llm_q4_0.gguf")

    # voices are encoded once here; afterwards the speech tokenizer and speaker encoder are unused
    model.add_zero_shot_spk(PROMPT_TEXT, f"{COSYVOICE_DIR}/asset/zero_shot_prompt.wav", "default")
    for wav in sorted(glob.glob(f"{VOICES_DIR}/*.wav")):
        name = os.path.splitext(os.path.basename(wav))[0]
        if not os.path.exists(f"{VOICES_DIR}/{name}.txt"):
            print(f"voice {name}: skipped, no {name}.txt transcript next to the wav", flush=True)
            continue
        with open(f"{VOICES_DIR}/{name}.txt") as f:
            text = f.read().strip()
        model.add_zero_shot_spk(text if "<|endofprompt|>" in text else INSTRUCT + text, wav, name)
        print(f"voice {name}", flush=True)
    model.frontend.speech_tokenizer_session = model.frontend.campplus_session = None

    mm = model.model
    llm_job, token2wav = mm.llm_job, mm.token2wav
    llm_lock, t2w_lock = FifoLock(), FifoLock()

    def staged_llm_job(*a, **k):
        with llm_lock:
            return llm_job(*a, **k)

    def staged_token2wav(*a, **k):
        with t2w_lock:
            out = token2wav(*a, **k)
        print(f"token2wav {steps.last}", flush=True)
        return out

    mm.llm_job, mm.token2wav = staged_llm_job, staged_token2wav
    return model


MODEL = load()
# loading leaves gigabytes of freed checkpoint copies in glibc's heap; hand them back to the system
gc.collect()
ctypes.CDLL("libc.so.6").malloc_trim(0)
GATE_HOLD = npu_cosyvoice.NpuGate(GATE)
app = FastAPI()


class Speech(BaseModel):
    input: str
    model: str = "cosyvoice3"
    voice: str = "default"
    response_format: str = "mp3"
    speed: float = DEFAULT_SPEED


def encode(audio, fmt):
    media, ff = FORMATS[fmt]
    if fmt == "pcm":
        return (np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes(), media
    if ff is None:
        buf = io.BytesIO()
        soundfile.write(buf, audio, MODEL.sample_rate, format=fmt.upper())
        return buf.getvalue(), media
    pcm = (np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes()
    r = subprocess.run(["ffmpeg", "-loglevel", "error", "-f", "s16le", "-ar", str(MODEL.sample_rate), "-ac", "1",
                        "-i", "pipe:0", *ff, "pipe:1"], input=pcm, capture_output=True, check=True)
    return r.stdout, media


@app.post("/v1/audio/speech")
def speech(req: Speech):
    if req.response_format not in FORMATS:
        raise HTTPException(400, f"response_format must be one of {sorted(FORMATS)}")
    if not req.input.strip():
        raise HTTPException(400, "empty input")
    if len(req.input) > 4096:
        raise HTTPException(400, "input is limited to 4096 characters")
    if not 0.5 <= req.speed <= 2.0:
        raise HTTPException(400, "speed must be between 0.5 and 2.0")
    voice = req.voice if req.voice in MODEL.frontend.spk2info else "default"
    t0 = time.perf_counter()
    out = MODEL.inference_zero_shot(req.input, "", "", zero_shot_spk_id=voice, stream=False, speed=req.speed)
    audio = torch.cat([o["tts_speech"] for o in out], 1)[0].float().numpy()
    body, media = encode(audio, req.response_format)
    took = time.perf_counter() - t0
    print(f"speech {len(audio) / MODEL.sample_rate:.2f}s audio in {took:.2f}s ({voice})", flush=True)
    return Response(body, media_type=media)


@app.post("/v1/audio/transcriptions")
async def transcriptions(request: Request):
    global STT_PENDING
    if not STT_URL:
        raise HTTPException(503, "no speech-to-text service configured (start the bundle with --profile stt)")
    if int(request.headers.get("content-length") or 0) > STT_MAX_BYTES:
        raise HTTPException(413, "audio request too large")
    if STT_PENDING >= STT_MAX_PENDING:
        raise HTTPException(429, "speech-to-text service busy")
    STT_PENDING += 1
    try:
        return await forward_transcription(request)
    finally:
        STT_PENDING -= 1


async def forward_transcription(request):
    chunks, size = [], 0
    async for chunk in request.stream():
        size += len(chunk)
        if size > STT_MAX_BYTES:
            raise HTTPException(413, "audio request too large")
        chunks.append(chunk)
    body = b"".join(chunks)
    fwd = urllib.request.Request(f"{STT_URL}/v1/audio/transcriptions", data=body, method="POST",
                                 headers={"Content-Type": request.headers.get("content-type", "")})

    def call():
        with GATE_HOLD.hold():
            try:
                with urllib.request.urlopen(fwd, timeout=120) as r:
                    return r.status, r.read(), r.headers.get("Content-Type", "application/json")
            except urllib.error.HTTPError as e:
                # Upstream exception bodies can contain deployment paths, URLs or secrets.
                # Preserve the HTTP status, but keep diagnostics on the upstream server.
                status = e.code
                e.close()
                message = {400: "invalid audio request", 413: "audio request too large",
                           422: "invalid audio request", 429: "speech-to-text service busy",
                           503: "speech-to-text service unavailable"}.get(status, "speech-to-text request failed")
                return status, json.dumps({"error": message}).encode(), "application/json"
            except OSError:
                return 502, json.dumps({"error": "speech-to-text service unavailable"}).encode(), "application/json"

    status, data, media = await run_in_threadpool(call)
    return Response(data, status_code=status, media_type=media)


@app.get("/v1/audio/voices")
def voices():
    return {"voices": [{"id": v, "name": v} for v in MODEL.frontend.spk2info]}


@app.get("/v1/models")
def models():
    return {"object": "list", "data": [{"id": "cosyvoice3", "object": "model"}] +
            ([{"id": "whisper", "object": "model"}] if STT_URL else [])}


@app.get("/health")
def health():
    return {"status": "ok", "stt": bool(STT_URL)}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=PORT)
