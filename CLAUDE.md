# Diagen Project Notes

## Model Distribution — Git LFS

All AI model weights ship in-repo. Two distribution paths:

**Server-side models (via Git LFS):** `models/llm/`, `models/audio2afan/`, `models/tts/`, `models/whisper/`.
LFS patterns in `.gitattributes`: `*.onnx`, `*.bin`, `*.pth`, `*.pt`, `*.gguf`, `*.npz`, `*.vrm`, `*.safetensors`, `models/**/tokenizer.model`.
After `git clone`, run `git lfs pull` to fetch weights (~2 GB). Verify with `node download-models.js` (no download; just an LFS-aware presence check that warns on pointer files).

**Browser demo (raw git blobs, NOT LFS):** GitHub Pages serves raw git blobs and does not resolve LFS pointers, so the browser ONNX model must stay as plain bytes. To stay under GitHub's per-file limit, `gh-pages-src/demo/model/onnx/*.onnx` is split into `≤99MB` `.part*` chunks. These paths carry explicit LFS exclusions in `.gitattributes` (`!filter !diff !merge -text`). The fetch interceptor in `gh-pages-src/demo/worker.js` reassembles the parts at load time.
The browser voice safetensors (`gh-pages-src/demo/voices/*.safetensors`) and `Cleetus.vrm` are also excluded from LFS for the same reason.

## Browser Demo Model — Non-Negotiable

The browser demo uses Qwen3.5-VL 0.8B abliterated with identity-override LoRA merged, quantized to q4 ONNX.
- Source: bobber/routangseng-qwen35-0.8b-abliterated-lora-onnx (HuggingFace)
- Quantization: q4 via MatMulNBitsQuantizer (decoder), fp16 (embed_tokens, WebGPU), q8 (vision_encoder)
- Delivery: raw git blobs split into ≤99MB chunks served from GitHub Pages. No LFS on this path — LFS would break Pages delivery.

## Dependencies

- `sttttsmodels` is a local file dep (`file:../sttttsmodels`). Clone it as a sibling directory or `npm install` will fail.
- `audio2afan` postinstall download: redundant now (weights ship in-repo via LFS). Safe to ignore or disable with `npm install --ignore-scripts`. `download-models.js` just verifies presence.

## Testing

Run `npm test` (vitest). 120 tests across 7 files. Pure functions are extracted into shared modules (`server-utils.mjs`, `animation-core.mjs`, `tokenizer.mjs`) that both source files and tests import.

## Quantize Workflow

To switch to Qwen2.5-0.5B abliterated (faster, roleplay-focused):
- Run `.github/workflows/quantize-model.yml` via GitHub Actions (workflow_dispatch)
- Requires `HF_TOKEN` secret if downloading gated models
- Workflow: downloads PyTorch weights → optimum-cli ONNX export → MatMulNBitsQuantizer q4f16 → splits into ≤99MB parts → commits model files + updates worker.js to main
- `*.py` is in .gitignore — Python helpers are inlined as workflow heredocs, not separate files

## STT — Whisper Speech-to-Text

The browser demo and Discord voice integration use **Whisper** (OpenAI) for speech-to-text transcription.

### Node.js / Discord Voice

**Implementation**: `discord-whisper.js` module using `@xenova/transformers` library.

**Model**: `Xenova/whisper-tiny` (39MB quantized ONNX)
- Lightweight inference-optimized version of Whisper
- Suitable for real-time Discord voice processing
- Runs fully on-device, no external API calls

**API**:
```javascript
import { transcribe } from './discord-whisper.js';

// Discord PCM is 48kHz 16-bit mono
const result = await transcribe(pcmBuffer, 48000);
// Returns: { text: string, confidence: number }
```

**Features**:
- Automatic resampling 48kHz → 16kHz (Whisper requirement)
- Model cached in memory after first download
- Concurrent call handling (promise-based singleton)
- Memory-safe chunked processing (30s max per inference)

**First Call**: Downloads model (~39MB) from HuggingFace on first transcribe() call. Cached thereafter.

Alternative models available: `Xenova/whisper-small` (74MB), `Xenova/whisper-base` (137MB), `Xenova/whisper-medium` (308MB). Change in `discord-whisper.js` initPipeline() if needed.

## TTS — Pocket TTS WASM with Voice Cloning

The browser demo uses **Pocket TTS** (Kyutai Labs) compiled to WebAssembly for fast, on-device text-to-speech.

### Built-in Voices
7 pre-recorded voices from Kyutai: alba, marius, javert, fantine, cosette, eponine, azelma. Voice embeddings loaded from HuggingFace (`kyutai/pocket-tts-without-voice-cloning/embeddings_v2/`).

### Custom Voice Cloning
Custom voices (cleetus, vampire) are pre-encoded as KV cache states in `.safetensors` format:

**Encoding pipeline (`ci/encode-voices/src/main.rs`):**
1. Downloads `kyutai/pocket-tts` model weights via HuggingFace hub
2. Reads WAV files, resamples to 24kHz mono (up to 10s)
3. Runs mimi encoder → voice embeddings
4. Runs `prompt_audio` → fills transformer KV caches
5. Saves 6 layers of k+v caches as `.safetensors`

**Browser loading:**
```javascript
// Built-in voices
model.add_voice(fetchBufWithCache(`${HF_BASE}/embeddings_v2/${name}.safetensors`))

// Custom voices (pre-encoded)
model.add_voice(fetchBufWithCache(`./voices/${name}.safetensors`))
```

### GitHub Actions: `encode-voices.yml`
Triggers when WAV files in `gh-pages-src/demo/voices/` change. Clones `LaurentMazare/xn`, builds `ci/encode-voices` Rust binary, runs it, commits `.safetensors`. Can also run manually via `workflow_dispatch`.

### Adding New Custom Voices
1. Record WAV (10-30 sec, any sample rate)
2. Place in `gh-pages-src/demo/voices/`
3. Update `manifest.json` with filename
4. Commit → workflow auto-encodes and pushes

### KV Cache Format
Each `.safetensors` contains pre-computed attention caches for 6 transformer layers:
```
transformer.layers.{i}.self_attn/cache
  Shape: [2, 1, seq_len, num_heads, head_dim] (k and v caches)
  Type: float32
transformer.layers.{i}.self_attn/current_end
  Shape: [1], position in cache
```

## Browser Worker — ONNX Loading Pitfalls

- **transformers.js filename construction**: with `dtype: 'q4f16'` (string) and `model_file_name: 'model_q4f16'`, transformers.js requests `model_q4f16_q4f16.onnx` (base + `_` + dtype). The fetch interceptor CHUNKS key must match this exact URL suffix.

- **Part file URL pattern**: chunk files are named `model_q4f16.onnx.part0` (the `.onnx` extension precedes `.part`). The `fetchChunked` URL must be `${stem}.onnx.part${i}`, not `${stem}.part${i}`. A wrong URL silently serves GitHub Pages 404 HTML.

- **HTML-poisoned cache**: GitHub Pages 404 responses are HTML (`<!DOCTYPE html>`) and can be large enough to pass a byte-size check. The cache bust must also check the first byte: `new Uint8Array(buf.slice(0,1))[0] === 0x3C` means HTML, delete it.

## TTS — Chatterbox Turbo (ResembleAI, ONNX via @huggingface/transformers)

Both server-side and browser demo now use **Chatterbox Turbo** (ResembleAI) for text-to-speech. Chatterbox is ONNX-based, eliminating the subprocess overhead of previous Python bridges.

### Node.js / Discord Voice

**Implementation**: `chatterbox-tts-bridge.js` module using `@huggingface/transformers` v4 (NOT v2 — v2 lacks ChatterboxModel).

**Speaker Pre-Encoding**:
Before any synthesis call, encode a reference voice once via `setRefVoice(wavPath)`:
```javascript
import { setRefVoice, synthesize, synthesizeStream } from './chatterbox-tts-bridge.js'

await setRefVoice('/path/to/voices/cleetus.wav')
const { audio, sampleRate } = await synthesize('hello world', _refPath, _refText, signal)
```

- `setRefVoice(wavPath)` decodes WAV → monoFloat32Array, calls `model.encode_speech(Tensor('float32', monoFloat32, [1, samples.length]))`, caches embedding
- `_refPath` and `_refText` parameters ignored (speaker pre-encoded via `setRefVoice`)
- Output: `{ audio: Float32Array, sampleRate: 24000 }`
- Streaming via `synthesizeStream(text, _refPath, _refText, onChunk?, signal?)`

**API Contract** (backward-compatible with old Qwen3 bridge):
```javascript
const { audio, sampleRate } = await synthesize(text, _unused, _unused, signal)
await synthesizeStream(text, _unused, _unused, (chunk, sr) => { /* play */ }, signal)
```

**Integration Points**:
- **Discord voice pipeline** (`speak-gate.js` SPEAKING stage): pipes LLM output through `synthesizeStream`, upmixes mono→stereo, pushes via `pushAudioFrame`
- **Web demo API** (`/api/generate` in `server.js`): one-shot `synthesize` for facial animation flow

**Observability**: `GET /debug/tts` returns `{ modelLoaded, speakerEncoded, loading }`

### Browser Demo

**Implementation**: `gh-pages-src/demo/worker.js` loads `@huggingface/transformers` v4.2.0 from CDN:
```
https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.2.0/+esm
```

**Speaker Encoding** (browser-side):
```javascript
const wavBuffer = await fetch('./voices/cleetus.wav').then(r => r.arrayBuffer())
const audioCtx = new OfflineAudioContext(1, 48000, 48000)
const audioBuffer = await audioCtx.decodeAudioData(wavBuffer)
const monoF32 = audioBuffer.getChannelData(0)
await model.encode_speech(new Tensor('float32', monoF32, [1, monoF32.length]))
```

**Sample Rate**: 24000 Hz (Chatterbox native). Output Float32Array.

**Why Chatterbox**: ONNX (no subprocess), lightweight inference via transformers.js, same API contract as old Qwen3 bridge for drop-in replacement, browser-native (no Python runtime needed).

**Backward Compatibility**: `qwen3-tts-bridge.js` and `qwen3_tts_server.py` preserved as B1 alternative if Chatterbox needs to be swapped back.
## node-llama-cpp — GPU Detection & Invocation Pitfall

**Critical diagnostic lesson (witnessed 2026-04-22)**:

False diagnosis led to a multi-week Rust rewrite plan. **Root cause**: When invoking node-llama-cpp via `node --input-type=module -e "import { getLlama } ... "`, the flag propagates to child processes. node-llama-cpp uses `child_process.fork(testBindingBinary.js)` to probe CUDA addon availability. The child inherits `--input-type=module`, which is invalid for file execution (only valid for `--eval`/`--print`/STDIN). Child exits with `ERR_INPUT_TYPE_NOT_ALLOWED`. node-llama-cpp interprets this as "CUDA failed", falls back to Vulkan (same error), then to CPU — silently.

**Always invoke via a real `.mjs` file**, not `node --input-type=module -e`. Example:
```javascript
// probe.mjs
import { getLlama } from 'node-llama-cpp';
const llama = await getLlama();
console.log('GPU:', llama.getGpu());
```
Then: `node probe.mjs`

**Performance impact** (RTX 3060 Laptop, CUDA v12.6):
- Wrong invocation: 12–28s (CPU fallback), grammar-constrained generation 12–28s per call
- Correct invocation: 2.3s getLlama(), 128ms warm generation (100× faster)

**Lesson**: When probing packages using `child_process.fork()` on their own files (addon tests, binding probes), never use `node --input-type=module -e`. The flag propagates to children where it's invalid and causes silent fallbacks.

## Discord Bot Integration

Diagen includes optional Discord bot support for text and voice interactions.

### Setup

1. Create a Discord bot at https://discord.com/developers/applications
2. Copy the bot token and add to `.env`:
   ```
   DISCORD_TOKEN=your_token_here
   ```
3. Invite the bot to your server with `bot` scope and these permissions: Send Messages, Read Message History, Connect, Speak, Use Voice Activity
4. Start the server normally — Discord bot initializes automatically if `DISCORD_TOKEN` is set

### Features

**Text Commands** (`!diagen <prompt>`):
- Responds in any channel where bot has message permissions
- Automatically splits responses >2000 chars into multiple messages
- Ignores bot messages and DMs

**Voice (processing pipeline)**:
- Listen to users in voice channels via onUserAudio callback
- Process audio through discord-voice-processor pipeline
- Synthesize responses and send to Discord voice connection
- Full end-to-end pipeline: transcribe → generate → synthesize → resample

### Architecture

Discord voice uses **dispipe** npm package (low-level Discord gateway + UDP wrapper):
- `dispipe/client`: joinDiscordVoice(), subscribeToSpeaker(), leaveVoice()
- `dispipe/voice`: initVoicePlayer(), pushAudioFrame()

Integration modules:
- `discord-handler.js` — Initializes dispipe client, manages voice connections, coordinates with VAD
- `discord-vad.js` — Voice Activity Detection: stereo downmix, RMS thresholding (0.01), silence flush (1.5s)
- `discord-voice-processor.js` — Audio pipeline: transcribe → generate → synthesize → resample → pushAudioFrame
- `server.js` — API endpoints for Discord control
  - `POST /api/discord/voice/connect` — join voice channel
  - `POST /api/discord/voice/disconnect` — leave voice channel
  - `POST /api/discord/message` — send message to channel

### Voice Channel Selection

**Command**: `!join <channel-id>`
- Stores selected guild and channel IDs in module state
- Calls connectToVoiceChannel() to join the voice channel
- Provides user feedback on join attempt

**Module State**:
```javascript
currentChannelState = { guildId: null, channelId: null }
```

**API Functions**:
- `handleJoinCommand(guildId, channelId)` — async function to store channel state and connect
- `getCurrentChannelState()` — getter returning copy of stored channel state
- `getDebugState()` — getter returning debug state object (see Observability below)

### Audio Output (dispipe)

Audio output from discord-voice-processor.js is sent via `pushAudioFrame(f32)` from `dispipe/voice`.

**Function**: `pushAudioFrame(Float32Array)` in dispipe/voice package

Sends Float32Array mono audio to active Discord voice connection. dispipe handles internal Opus encoding and UDP transmission.

**Call site**: `speak-gate.js` audio sink (set via `setAudioSink()` from `discord-vad.init()`), upmixed mono→stereo before push.

### Observability

**Debug Endpoint**: `GET /debug/discord`

Returns real-time Discord bot state as JSON:
```json
{
  "connected": boolean,
  "guildId": string | null,
  "channelId": string | null,
  "lastError": string | null,
  "messageCount": number,
  "processingQueue": array,
  "audio": {
    "audioQueueLength": number,
    "totalAudioFramesSent": number,
    "lastSendTimestamp": number | null,
    "lastSendError": { message: string, timestamp: number } | null,
    "queueHistory": array
  }
}
```

This permanent, queryable endpoint provides complete visibility into:
- Connection status (whether bot is logged into Discord)
- Current voice channel selection (guild and channel IDs)
- Last error encountered (if any)
- Message count for monitoring activity
- Active processing queue for debugging
- Audio send metrics: queue length, total frames sent, last send timestamp, errors

Query via curl or monitoring tools: `curl http://localhost:8080/debug/discord`

### Speak-Gate State Machine — Voice Orchestration

**Module**: `speak-gate.js` — single shared 5-state machine driving the Discord voice loop. Replaces the previous utterance-triggered `processTranscript` flow.

**States**:
```
LISTENING ─[whisper word]→ WAITING (1s debounce)
WAITING   ─[whisper word]→ WAITING (re-arm 1s)
WAITING   ─[1s silent]→    GATING
GATING    ─[whisper word]→ abort, → WAITING
GATING    ─[NO]→           LISTENING
GATING    ─[YES]→          ANSWERING
ANSWERING ─[whisper word]→ abort, → WAITING
ANSWERING ─[done]→         SPEAKING
SPEAKING  ─[whisper word]→ abort, history(partial), → WAITING
SPEAKING  ─[done]→         history(full), → LISTENING
```

**Two-stage LLM**:
1. **GATING** — single grammar-constrained call returning `YES` or `NO`. Grammar `root ::= "YES" | "NO"` built via `buildGrammar()` from `llm-llamacpp.js` (must use the same `getLlama()` instance that loaded the model — `LlamaGrammar` instance must match the session's instance, otherwise `node-llama-cpp` throws). The gating prompt asks: should the bot speak now?
2. **ANSWERING** — full LLM call only fired when GATING returned YES, then piped into `synthesizeStream` from `qwen3-tts-bridge.js`.

**Per-stage AbortController + timeouts** (env-tunable): `GATE_TIMEOUT_GATING_MS=5000`, `GATE_TIMEOUT_ANSWER_MS=15000`, `GATE_TIMEOUT_SPEAKING_MS=30000`. A whisper word arriving during any post-LISTENING stage aborts the in-flight stage and snaps to WAITING.

**History accounting**:
- User words: each whisper update from a speaker collapses into the last entry if it's the same speaker, otherwise appends. Tagged `[username]`.
- Bot speech: written **only if at least one TTS chunk reached the audio sink**. On clean SPEAKING completion, the full text is committed; on whisper-mid-speak abort, an estimated partial (proportional to chunks played) is committed. Abort before any audio = nothing in history.

**Inputs**:
```javascript
import { noteWhisperWord, setRefVoice, setCharacterCardPrompt, setAudioSink, getDebugSnapshot } from './speak-gate.js'

setAudioSink((monoF32, _text) => { /* upmix mono→stereo, pushAudioFrame */ })
setRefVoice('/path/to/voices/cleetus.wav', '<transcript of cleetus.txt>')
setCharacterCardPrompt('You are Cleetus...')
noteWhisperWord({ userId, username, text })
```

`noteWhisperWord` filters wordless / sentinel inputs (`[BLANK_AUDIO]`, `*music*`, `(upbeat music)`, whitespace) before re-arming the debounce timer. Sentinels with all three bracket styles — `[...]`, `*...*`, `(...)` — are caught by both `whisper-stream.js:isSentinel()` and `speak-gate.js:isWordlessOrSentinel()`.

### Discord VAD — RMS Gate, Not Subscription Management

**Module**: `discord-vad.js` — receives stereo PCM frames from `dispipe/voice` per active speaker, downmixes to mono, applies AGC (`TARGET_RMS=0.15`), and **gates the whisper feed by RMS** (`VAD_ACTIVE_RMS=0.005` — frames below this floor are not pushed to whisper-stream). dispipe v1.0.1 has no `unsubscribeFromSpeaker`, so the gate is at the data-receiving callback, not at subscription level.

`onPartial` and `onStable` callbacks from `whisper-stream.js` both feed `speak-gate.noteWhisperWord` — partials are how we detect a speaker is still mid-utterance.

The bot's own TTS audio is masked from re-entering whisper via `_botSpeakingUntil` (set when the audio sink writes a chunk; subsequent inbound frames are skipped during that window).

### Whisper Stream — Warm Worker Pool, Per-User Sessions

**Module**: `whisper-stream.js` — single warm `@xenova/transformers` Whisper pipeline shared across all per-user sessions. Sessions are a `Map` keyed by userId; each holds the rolling PCM buffer, debounced re-transcription scheduling (200ms), and stability detection (350ms). **No spawn/teardown** of workers per speaker — only `clear(userId)` to drop accumulated audio.

### Observability — `/debug/speak-gate`

`GET /debug/speak-gate` returns a live snapshot:
```json
{
  "state": "LISTENING|WAITING|GATING|ANSWERING|SPEAKING",
  "msInState": 1234,
  "debounceArmed": true,
  "msUntilTick": 800,
  "activeAbortReason": "in-flight" | null,
  "lastDecision": { "decision": "YES", "at": 1700000000000 },
  "history": [...],
  "activeSpeakers": [{ "userId", "username", "lastWordAt", "lastText" }],
  "vadSpeakers": [{ "userId", "gain", "lastActiveAt", "skipped" }],
  "metrics": { "gateYes", "gateNo", "abortsByStage", "timeouts", "spoken" }
}
```

### Dependencies

Added: `discord.js`, `@discordjs/voice`, `prism-media`, `@xenova/transformers`, `dispipe`

### dispipe Audio Format — Critical Pitfalls

**subscribeToSpeaker emits stereo-interleaved Float32 at 48kHz:**
- Format: [L, R, L, R, ...] Float32Array, NOT mono
- Must downmix before Whisper STT: `mono[i] = (stereo[i*2] + stereo[i*2+1]) / 2`
- Implementation: discord-vad.js onPcmChunk() handler, lines 66-67

Why: Discord voice mix is stereo. Whisper requires mono 16kHz.

**pushAudioFrame expects stereo-interleaved Float32Array at 48kHz:**
- Input: Float32Array [-1.0 to 1.0], 48kHz, stereo [L,R,L,R,...] — NOT mono
- speak-gate's audio sink (set in `discord-vad.init`) upmixes mono→stereo before push: `const s=new Float32Array(mono.length*2); for(let i=0;i<mono.length;i++){s[i*2]=mono[i];s[i*2+1]=mono[i]}`
- dispipe encoder: channels=2, FRAME=960*2*2 bytes

Why: Opus encoder in dispipe/voice is stereo. Mono input → half-speed/wrong-pitch audio.

**VAD constants** (discord-vad.js):
- `VAD_ACTIVE_RMS = 0.005` (env-tunable) — frames below this floor are not pushed to whisper-stream (saves transcription on silence/background noise)
- `TARGET_RMS = 0.15`, `MAX_GAIN = 25`, `MIN_GAIN = 1`, `GAIN_ATTACK = 0.25` — automatic gain control toward target loudness
- `BOT_SPEAK_TAIL_MS = 250` — extra dead-time after bot's last audio chunk to prevent self-pickup

Why: Prevents sending empty audio and excessive fragmentation. Constants tuned empirically for natural speech.

**Event pattern for speaker subscription**:
```javascript
voiceReceiver.speaking.on('start', (userId) => {
  subscribeToSpeaker(userId, onPcmChunk)  // emit handler called with (userId, stereoFloat32)
})
```

### Reference Implementation

**webrig companion** (C:/dev/webrig/companion/index.js) uses identical dispipe pattern for Discord voice. Reference for dispipe API usage, stereo downmix logic, and VAD tuning.

## Testing — Discord Voice Pipeline

**Test File**: `test/discord-voice-pipeline.test.mjs` (67 lines, 4 tests)

Vitest suite verifying voice processing pipeline components:

1. **whisper-stt**: Validates Whisper STT pipeline accepts 48kHz PCM buffer from Discord
2. **tts-synthesis**: Validates TTS pipeline accepts text input and outputs float32 audio
3. **resampling-24k-to-48k**: Validates linear interpolation upsampling (24kHz → 48kHz)
4. **full-pipeline**: End-to-end integration test ensuring all pipeline stages connect without crashing

**Mock Audio**: 1-second 48kHz Int16Array buffer (48000 samples, 96KB). Uses sine wave pattern for realistic audio data.

**Real Imports**: Tests import actual `resampleAudio` from `server-utils.mjs` for witnessed resampling verification (not mocked).

**Run**: `npm test -- test/discord-voice-pipeline.test.mjs`

## Discord Context — Per-Channel Message History

**Module**: `discord-context.js` (47 lines)

In-memory context store for Discord voice interactions. Maintains per-guild/channel message history for stateful response generation.

**Exports**:
- `addMessage(guildId, channelId, userId, role, text)` — Append message with timestamp
- `getContext(guildId, channelId)` — Retrieve last 20 messages (or all if fewer)
- `clearContext(guildId, channelId)` — Delete all messages for a channel

**Storage Model**:
- Map-based: keyed by `"${guildId}:${channelId}"` (guild/channel isolation)
- FIFO queue per key: max 50 messages, drops oldest when exceeded
- Each message: `{ userId, role, text, timestamp }`

**Integration Points**:
- Optional: Import into `discord-handler.js` to track !diagen commands and responses
- Optional: Call in `disconnectFromVoiceChannel()` to clear history on channel exit
- Optional: Expose via `getDebugState()` for observability endpoint

**Use Case**: Enable multi-turn context for future LLM-based Discord responses. Store user prompts and bot replies for context window in subsequent message processing.

**No Dependencies**: Pure JavaScript, Map-based data structure, no external packages.

