# Voice bundle

Text-to-speech and speech-to-text for any chat model, on one Ryzen AI NPU, behind the OpenAI audio
API. Point a client that speaks it (Open WebUI, LibreChat, Home Assistant, your own code) at
`http://localhost:8000/v1` and whatever LLM it talks to gets a voice.

| endpoint | what runs |
|---|---|
| `POST /v1/audio/speech` | CosyVoice3: flow DiT int8 on the NPU, LM backbone in llama.cpp on 2-4 CPU cores |
| `POST /v1/audio/transcriptions` | [whisper-xdna](https://github.com/drakosha/whisper-xdna) large-v3-turbo: encoder on the NPU, decoder on the CPU |

The two share the NPU: whisper-xdna gets 5 of XDNA1's 6 hardware contexts and the TTS one, and
while a transcription runs the TTS pauses at the next DiT layer, so recognition is not slowed by
speech being synthesized at the same time.

## Run

Needs an XDNA1 machine (Ryzen AI 7040/8040) with the amdxdna driver (Linux 6.14+), Docker with
the compose plugin, ~18 GB of disk and ~8 GB of free RAM.

```sh
export RENDER_GID=$(getent group render | cut -d: -f3)
docker compose -f voice/compose.yml run --rm setup        # once: downloads ~4 GB
docker compose -f voice/compose.yml --profile stt up -d   # or without --profile stt for TTS only
```

The first start of `whisper` downloads its checkpoint (~1.6 GB) and compiles its NPU overlays,
which takes several minutes; `docker compose -f voice/compose.yml logs -f whisper` shows progress.

```sh
curl -s localhost:8000/v1/audio/speech -H 'Content-Type: application/json' \
    -d '{"input": "Hello from the NPU.", "voice": "default", "response_format": "mp3"}' -o hello.mp3
curl -s localhost:8000/v1/audio/transcriptions -F file=@hello.mp3 -F model=whisper
```

In Open WebUI: Admin Settings > Audio, both engines "OpenAI", URL `http://<host>:8000/v1`, any
API key, TTS model `cosyvoice3`, STT model `whisper`. (If Open WebUI runs in a container, `<host>`
is this machine's LAN address or `host.docker.internal`, not `localhost`.)

There is no authentication. By default the port is open on every interface of the host; set
`VOICE_BIND=127.0.0.1` to keep it local, or put a reverse proxy with authentication in front
before exposing it beyond a trusted network.

## Voices

`default` is the reference recording that ships with CosyVoice. To add one, put `<name>.wav`
(5-15 s of clean speech) and `<name>.txt` (exactly what is said) into the `voices` folder of the
`data` volume and restart `voice`. Only use recordings you have the right to clone. CosyVoice3
speaks Chinese, English, Japanese, Korean and several more; a voice keeps the accent of its
reference, so an English reference reads English best.

## Settings

| variable | default | |
|---|---|---|
| `VOICE_PORT` | 8000 | host port |
| `VOICE_BIND` | 0.0.0.0 | host interface |
| `STT_LANGUAGE` | auto | whisper's language when the request does not name one; `auto` detects it, and Chinese gets punctuation and simplified characters |
| `NPU_DEVICE` | npu1 | `npu2` builds the overlay for XDNA2 (Ryzen AI 300); untested, see the main README |

`STT_URL` points the transcription endpoint at another whisper-compatible server instead.

`setup` downloads the CosyVoice3 checkpoint from ModelScope and the prebuilt int8 weights, LM GGUF and
NPU kernel from [Hugging Face](https://huggingface.co/billchen42/npu-cosyvoice) (checksummed). With
`PREBUILT_REPO=` (empty), or if the checkpoint differs from the tested one, it builds those three locally
instead (~15 min of CPU); they are deterministic either way.

If the NPU fails to resume from runtime suspend under memory pressure, see Notes in the main README.
