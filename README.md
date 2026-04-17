# diagen

Real-time AI voice bot for Discord. Listens to users in voice channels, transcribes speech (Whisper), generates responses (Ollama LLM), synthesizes speech (OmniVoice), and plays back in the channel.

## Architecture

```
Discord voice → dispipe/client → discord-vad.js → Whisper STT → Ollama LLM → OmniVoice TTS → dispipe/voice → Discord
```

**Modules:**
- `discord-handler.js` — bot lifecycle, voice channel join, message commands
- `discord-vad.js` — RMS VAD, utterance buffering, mono→stereo upmix for dispipe, pipeline dispatch
- `discord-voice-processor.js` — STT → LLM → TTS pipeline, returns Float32Array at 48kHz
- `discord-whisper.js` — Whisper STT via @xenova/transformers
- `omnivoice-tts-bridge.js` — OmniVoice Python subprocess bridge
- `omnivoice_tts_server.py` — Python TTS server (spawned via uv run)
- `llm-ollama.js` — Ollama LLM integration

## Setup

```bash
npm install
cp .env.example .env
# Edit .env: DISCORD_TOKEN, GUILD_ID, CHANNEL_ID, OLLAMA_HOST
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
