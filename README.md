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
npm install
cp .env.example .env
# Edit .env: DISCORD_TOKEN, GUILD_ID, CHANNEL_ID, LLAMA_MODEL_PATH
node server.js
```

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
