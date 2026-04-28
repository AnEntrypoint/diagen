# AGENTS.md — Non-Obvious Caveats

## Dependencies

- **dispipe is on npm** (`dispipe@^1.0.2`). Do NOT downgrade to `file:../dispipe` — under bun that silently produces an empty `node_modules/dispipe` directory with EPERM, and the only surface symptom is `Cannot find module 'dispipe/client'` at `discord-handler.js` import time. No install error is printed.

## First-Boot / Server Startup

- **Fresh-checkout boot blocks 10-15 min on a hidden Chatterbox download.** `ResembleAI/chatterbox-turbo-ONNX` (~5-8GB across `embed_tokens` / `speech_encoder` / `language_model` / `conditional_decoder` q4f16 `.onnx` + `.onnx_data`) is fetched on first run. Cache lives at `node_modules/@huggingface/transformers/.cache/` (NOT `~/.cache/huggingface/hub`). Default logs only print `[chatterbox] loading processor + model (cache miss)...` for the whole window — no progress.
- `WARMUP_TTS=false` does NOT skip this download: `loadVoiceEmbedding` calls `setRefVoice` which calls `ensureProcessorOnly()` unconditionally.
- `server.js start()` gates `app.listen()` AND Discord login behind TTS warmup, so the HTTP server and bot are unreachable until cache warms. For offline validation, prefer: (a) check `node_modules/dispipe/src/bot/client.js` exists, (b) import `discord-*` modules in isolation, (c) run vitest.

## Learning audit

- 2026-04-28: initial creation; 0 items checked, 0 removed, 0 refined.
