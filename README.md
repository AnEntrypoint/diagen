# diagen

Real-time AI voice bot for Discord. Listens to users in voice channels, transcribes speech (Whisper), generates responses (llama.cpp LLM), synthesizes speech (OmniVoice), and plays back in the channel.

## Architecture

```
Discord voice → dispipe/client → discord-vad.js → Whisper STT → llama.cpp LLM → OmniVoice TTS → dispipe/voice → Discord
```

**Modules:**
- `discord-handler.js` — bot lifecycle, voice channel join, message commands
- `discord-vad.js` — RMS VAD, utterance buffering, mono→stereo upmix for dispipe, pipeline dispatch
- `discord-voice-processor.js` — STT → LLM → TTS pipeline, returns Float32Array at 48kHz
- `discord-whisper.js` — Whisper STT via @xenova/transformers
- `omnivoice-tts-bridge.js` — OmniVoice Python subprocess bridge
- `omnivoice_tts_server.py` — Python TTS server (spawned via uv run)
- `llm-llamacpp.js` — llama.cpp LLM integration via node-llama-cpp (in-process GGUF)

## Setup

```bash
git lfs install
git clone https://github.com/AnEntrypoint/diagen.git
cd diagen
git lfs pull             # pulls model weights (~2GB)
npm install
cp .env.example .env
# Edit .env: DISCORD_TOKEN, GUILD_ID, CHANNEL_ID
node server.js
```

## Models

All model weights ship in-repo via **Git LFS** under `models/`. Run `git lfs pull` after cloning.

| Dir | Contents | Size |
|---|---|---|
| `models/llm/*.gguf` | llama3.2-1b Q8_0 GGUF | ~1.3 GB |
| `models/audio2afan/` | ONNX + NPZ blendshape model | ~455 MB |
| `models/tts/` | Mimi + flow ONNX, tokenizer | ~199 MB |
| `models/whisper/` | Xenova/whisper-base ONNX | ~150 MB |
| `models/omnivoice/` | HF cache (gitignored; auto-downloaded on first TTS call) | ~500 MB |

Override LLM path with `LLAMA_MODEL_PATH`; defaults to first `.gguf` in `models/llm/`.

## Environment

| Variable | Description |
|---|---|
| `DISCORD_TOKEN` | Discord bot token |
| `GUILD_ID` | Guild ID to auto-join |
| `CHANNEL_ID` | Voice channel ID to auto-join |
| `PORT` | HTTP server port (default 8080) |
| `WARMUP_TTS` | Set to `false` to skip TTS warmup |
| `LLAMA_MODEL_PATH` | Path to GGUF model file (default: ollama blob for llama3.2:1b) |
| `LLAMA_CONTEXT_SIZE` | Context size tokens (default 2048) |
| `LLAMA_GPU_LAYERS` | Number of layers to offload to GPU (default: auto) |

## HTTP API

| Endpoint | Method | Description |
|---|---|---|
| `/api/generate` | POST | Text → audio + blendshape animation |
| `/api/chat` | POST | Text → LLM response |
| `/api/discord/voice/connect` | POST | Join voice channel |
| `/api/discord/voice/disconnect` | POST | Leave voice channel |
| `/api/discord/message` | POST | Send text message |
| `/debug/discord` | GET | Bot state inspection |

## Discord Commands

- `!join <channel-id>` — join a voice channel
- `!diagen <prompt>` — text chat with the bot

## Testing

```bash
node test.js
```
