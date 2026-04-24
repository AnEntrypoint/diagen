
## [unreleased]
- feat(browser): drop local Chatterbox TTS implementation — demo/tts-worker.js is now a thin wrapper over `streamtts@latest` (loaded from esm.sh), calling `ChatterboxSDK.configure({ modelBasePath })` pointed at `raw.githubusercontent.com/AnEntrypoint/streamtts/models/`. The streamtts package ships the SDK + worker + chunk-reassembly fetch interceptor; diagen no longer maintains its own copy.
- chore(ci): remove Chatterbox model download + chunking step from pages.yml. The model lives on streamtts's `models` branch (rebuilt on HF upstream change by build-model.yml there); diagen's gh-pages deploy is now a small orphan commit of just the demo + landing markup.
- fix(ci): pages.yml deploy step now publishes gh-pages-src as an orphan branch with force-push. Prior approach checked out main's gh-pages, overlaid files, and pushed — this leaked server source files into the branch and hit non-fast-forward rejections when runs raced.
- fix(pages): vendor design-system fonts (Archivo, Archivo Narrow, JetBrains Mono, Space Grotesk) to gh-pages-src/design/vendor/fonts/ + fonts.css; link from both index.html and demo/index.html so ff-display / ff-mono resolve to real webfonts instead of falling back to Times New Roman on the landing page.
- fix(ci): pages.yml snapshot_download now uses allow_patterns to fetch only q4 (wasm) + q4f16 (webgpu) ONNX variants. Prior full snapshot was ~4.3GB and caused git push HTTP 500 during gh-pages deploy. Chunker walks the onnx/ dir for any *.onnx_data > 99MB (no longer hardcoded to fixed filenames, so variant-suffixed files are handled automatically).
- feat(pages): migrate gh-pages-src to design system — sync colors_and_type.css + app-shell.css from c:/dev/design; both index.html and demo/index.html now use .app/.app-topbar/.app-main/.app-status shell; drop var(--border) (incompatible with border:0 reset); update tts receipt entry to chatterbox turbo
- feat(ci): pages.yml downloads ResembleAI/chatterbox-turbo-ONNX at build time via huggingface_hub; splits 3 oversized .onnx_data files (155-175MB) into ≤99MB .part* chunks; writes chunks.json manifest. Browser demo loads models locally with no CDN dependency at runtime.
- feat(browser): tts-worker.js sets env.localModelPath + env.allowRemoteModels=false; fetch interceptor reads chunks.json and reassembles .part* chunks transparently before ONNX runtime sees the data
- feat: replace Qwen3-TTS Python bridge with Chatterbox Turbo (ResembleAI/chatterbox-turbo-ONNX) via @huggingface/transformers v4 — no Python subprocess, Node.js-native ONNX inference. New chatterbox-tts-bridge.js: pre-encodes speaker WAV once at setRefVoice(), synthesize/synthesizeStream API matches old bridge contract. speak-gate.js and server.js updated.
- feat(browser): replace Pocket TTS WASM worker with Chatterbox Turbo WebGPU/WASM worker — @huggingface/transformers v4 from CDN, voice cloning from voices/*.wav, streams audio_chunk messages per sentence chunk, same app-tts.js protocol
- fix: whisper-stream.js isSentinel() now filters parenthesized sentinel outputs like "(upbeat music)" alongside [...] and *...* patterns. Prevents sentinel annotations from triggering speak-gate state transitions.
- feat: replace utterance-triggered processTranscript with 5-state speak-gate machine (LISTENING/WAITING/GATING/ANSWERING/SPEAKING). Whisper words debounce 1s into a grammar-constrained YES/NO gating LLM call; YES fires the answering LLM then streams TTS. Any whisper during a post-LISTENING stage aborts back to WAITING. Bot history written only when at least one TTS chunk played.
- new: speak-gate.js (188L, dispatch-table state transitions, per-stage AbortController, env-tunable timeouts)
- chore: discord-vad.js rewritten — RMS-gates whisper feed at pushFrame (dispipe has no unsubscribe), drops handleUtterance/processTranscript/preamble/speculative/interruption-resume plumbing
- chore: discord-voice-processor.js shrunk to 67L — config-only, forwards setVoiceEmbedding/setCharacterCard to speak-gate
- new: llm-llamacpp.js exposes buildGrammar() so consumers share the model's llama instance (mismatched instances throw)
- new: GET /debug/speak-gate observability endpoint
- feat: swap Pocket TTS for Qwen3-TTS-12Hz-0.6B via faster-qwen3-tts (CUDA-graph streaming, ~700-850ms first chunk, ~1.3x RT warm)
- new: qwen3-tts-bridge.js + qwen3_tts_server.py (Node↔Python subprocess, identical synthesize/synthesizeStream contract to prior bridges)
- chore: delete pocket-tts-bridge.js, pocket_tts_server.py, omnivoice-tts-bridge.js, omnivoice_tts_server.py
- fix: on voice close code 4017, force gateway shard reconnect (recover=0) to get fresh session_id instead of waiting 10s with stale session

## 2026-04-17
- Ported Discord voice to dispipe/client + dispipe/voice packages
- Added discord-vad.js: stereo→mono downmix, RMS VAD, utterance buffering
- discord-handler.js: voiceReceiver.speaking.on('start') event-based speaker subscription
- discord-voice-processor.js: returns Float32Array (48kHz) directly to pushAudioFrame
- Added test.js: VAD downmix, RMS threshold, PCM encoding, resample ratio assertions
# Changelog

## [Unreleased]

### Added
- OmniVoice Python TTS backend integration for Discord voice and server-side synthesis
  - omnivoice-tts-bridge.js: Node.js subprocess bridge for OmniVoice model
  - omnivoice_tts_server.py: Python inference server with stdin/stdout JSON IPC
  - Support for 600+ languages via OmniVoice multilingual training
  - Voice cloning via reference audio (ref_audio + ref_text parameters)
- Comprehensive OmniVoice documentation in CLAUDE.md
  - Setup instructions for Windows and Linux
  - Architecture overview and integration points
  - Voice cloning workflow with examples
  - Subprocess lifecycle and error handling

### Changed
- discord-voice-processor.js: Updated to use OmniVoice bridge instead of webtalk
  - Calls omnivoice-tts-bridge.synthesize() for TTS synthesis
  - Passes cleetus.wav as voice reference for consistent character voice
  - Output format unchanged: 24kHz float32 audio
- server.js: Updated /api/generate endpoint to use OmniVoice for synthesis
  - Changed from ttsOnnx/webtalk to OmniVoice backend
  - Maintains same input/output format for compatibility
  - Voice reference path set during server startup

### Removed
- Removed webtalk TTS dependency from package.json
- Removed sharp native module dependency (no longer needed)
- Eliminated build tool requirement for Windows compatibility

### Fixed
- Resolved Windows compatibility issues by removing native module dependencies
- Enabled cross-platform deployment without build tools

---

Session: OmniVoice TTS Integration
Date: 2026-04-15
Items Completed: 8/8
- create-omnivoice-tts-bridge
- create-omnivice-tts-python-server
- update-discord-voice-processor
- remove-webtalk-dependencies
- update-server-startup
- web-demo-tts-unchanged
- omnivoice-windows-docs
- replace-webtalk-with-omnivoice (parent task)

## [2026-04-17] Discord voice fixes

- fix-mono-stereo-upmix: upmix mono Float32 to stereo before pushAudioFrame (dispipe encoder is 2ch)
- fix-tts-sample-rate: synthesize() returns { audio, sampleRate }, processor uses actual rate for resample
- fix-omnivoice-line-limit: extract Python TTS server to omnivoice_tts_server.py, bridge now 100L
- fix-whisper-comments: removed all comments from discord-whisper.js
- delete discord-voice-player.js: orphaned, dispipe/voice is authoritative
