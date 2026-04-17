
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
