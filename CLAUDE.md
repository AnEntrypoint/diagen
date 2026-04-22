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

## TTS — Server-Side: Qwen3-TTS via faster-qwen3-tts

Discord voice and server-side text-to-speech use **Qwen3-TTS-12Hz-0.6B-Base** (Apache 2.0, Alibaba/QwenLM) wrapped by the **faster-qwen3-tts** community package (CUDA graph capture for low-latency streaming). Browser demo is unrelated and stays on Pocket TTS WASM (no browser runtime exists for Qwen3-TTS).

### Setup

Requires Python 3.10+, NVIDIA GPU with CUDA, PyTorch 2.5.1+. Installation:
```bash
pip install -U faster-qwen3-tts
```

On first synthesis call, downloads ~2.1GB of weights from HuggingFace to `~/.cache/huggingface/hub/models--Qwen--Qwen3-TTS-12Hz-0.6B-Base/`. Subsequent process starts reuse the cache.

### Architecture

**Node.js→Python bridge**: `qwen3-tts-bridge.js` spawns a persistent Python subprocess (`qwen3_tts_server.py`) speaking JSON over stdin/stdout.

```javascript
import { synthesize, synthesizeStream } from './qwen3-tts-bridge.js'

// One-shot — returns { audio: Float32Array, sampleRate: 24000 }
const { audio, sampleRate } = await synthesize('hello world', refAudioPath, refText, signal)

// Streaming — onChunk fires per audio chunk during generation
await synthesizeStream(text, refAudioPath, refText, (chunk, sr) => { /* play */ }, signal)
```

The bridge serializes calls via an internal queue chain, supports AbortSignal-driven cancellation, and reconnects on subprocess exit.

### Voice Cloning

Voice clone REQUIRES BOTH `ref_audio` (WAV path) AND `ref_text` (transcript of the reference). The Python server auto-loads a sidecar `.txt` next to the ref WAV — `voices/cleetus.wav` pairs with `voices/cleetus.txt`. Set defaults via env:

- `QWEN3_TTS_DEFAULT_REF` — WAV path (default: `voices/cleetus.wav`)
- `QWEN3_TTS_DEFAULT_REF_TEXT` — overrides sidecar
- `QWEN3_TTS_LANGUAGE` — default `English`
- `QWEN3_TTS_CHUNK_SIZE` — streaming chunk size (default 12)
- `QWEN3_TTS_PYTHON` — interpreter (default `python`)
- `QWEN3_TTS_STARTUP_MS` — startup timeout (default 1200000 to allow first download)
- `QWEN3_TTS_SYNTH_MS` — per-synth timeout (default 120000)

### Performance (RTX 3060 Laptop GPU, witnessed)

- Process start (cached weights): ~23s + first call ~13–15s (one-time CUDA graph capture for predictor + talker)
- Warm streaming: first chunk ~700–850ms, total ~1.3× realtime
- Warm one-shot: ~1.8× realtime
- Output: 24000 Hz Int16 PCM (decoded to Float32 in the bridge)

The subprocess server prints model warmup status to stdout; the server permanently rebinds `sys.stdout` to `sys.stderr` after startup so library prints do not poison the JSON IPC channel.

### Integration Points

- **Discord voice pipeline** (`discord-voice-processor.js`): transcribe user audio → generate text → `synthesizeStream` → resample 24k→48k → `pushAudioFrame`
- **Preamble cache** (`preamble-cache.js`): pre-renders short interjections at startup for instant playback
- **Web demo API** (`/api/generate` in `server.js`): one-shot `synthesize` for facial animation flow

### Switching back to a different engine

If `faster-qwen3-tts` breaks (it is a community fork), the official `qwen-tts` package implements the same `Qwen3TTSModel` API but lacks the streaming wrapper — `generate_voice_clone_streaming` would have to be replaced with one-shot `generate_voice_clone` calls in `qwen3_tts_server.py`. The bridge contract stays unchanged.

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

**Call site**: discord-vad.js handleUtterance(), line 51

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

### Voice Audio Processing Pipeline

**Module**: `discord-voice-processor.js` — Complete end-to-end audio processing pipeline for Discord voice interactions.

**Pipeline Flow**:
```
48kHz PCM Input → Transcribe → Generate → Synthesize → Resample → 48kHz Float32 Output
     (16-bit)       (STT)      (text)      (24kHz)      (24→48k)   (dispipe expects)
```

**API**:
```javascript
import { processUserAudio, setVoiceEmbedding } from './discord-voice-processor.js';

// Initialize with voice embedding (call once after loading voice)
setVoiceEmbedding(voiceReferencePath);

// Process user audio through complete pipeline
const pcmOutput = await processUserAudio(pcmBuffer, 48000, userId);
// Returns: Float32Array of 48kHz mono audio, ready for pushAudioFrame(pcmOutput)
```

**Step Details**:

1. **Transcribe** (`discord-whisper.js`): Converts 48kHz PCM to text using Whisper STT
   - Input: Buffer | Uint8Array, 48kHz mono 16-bit PCM
   - Output: { text: string, confidence: number }
   - Handles empty/silent audio gracefully

2. **Generate**: Template-based response generation from user text
   - Handles silence detection (returns default message if no speech)
   - Extensible for future LLM integration

3. **Synthesize** (`ttsOnnx`): Text-to-speech synthesis at 24kHz
   - Input: Response text, voice embedding tensor
   - Output: Float32Array at 24kHz
   - Uses server-loaded voice embedding for voice cloning

4. **Resample** (`server-utils.mjs`): Linear interpolation resampling from 24kHz to 48kHz
   - Ensures Discord voice channel compatibility
   - Preserves audio fidelity through smooth interpolation
   - Returns Float32Array at 48kHz

Output ready for pushAudioFrame(): Float32Array mono 48kHz. No further conversion needed.

**Error Handling**:
- All errors include context: step name, userId, input size/format
- Examples: `step=input userId=123`, `step=synthesize userId=456: <error details>`
- Propagates errors loudly (no silent fallbacks)

**Integration Points**:
- `discord-handler.js`: Calls processUserAudio in onUserAudio callback (async)
- `server.js`: Calls setVoiceEmbedding(voiceEmbedding) during startup to initialize
- Voice embedding loaded from voices/cleetus.wav (or configured voice file)

**Constraints**:
- Module is 135 lines, under 200-line limit
- Zero magic numbers (uses SAMPLE_RATE_DISCORD=48000, SAMPLE_RATE_TTS=24000 constants)
- Requires voice embedding to be set before processing audio
- TTS models must be loaded (done via ttsOnnx.loadModels() in server.js)

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
- processUserAudio returns mono — upmix before calling: `const s=new Float32Array(mono.length*2); for(let i=0;i<mono.length;i++){s[i*2]=mono[i];s[i*2+1]=mono[i]}`
- Called in discord-vad.js handleUtterance() after processUserAudio
- dispipe encoder: channels=2, FRAME=960*2*2 bytes

Why: Opus encoder in dispipe/voice is stereo. Mono input → half-speed/wrong-pitch audio.

**VAD (Voice Activity Detection) constants** (discord-vad.js):
- `SILENCE_THRESHOLD = 0.01` — RMS level threshold for speech/silence boundary
- `SILENCE_DURATION_MS = 1500` — Flush buffer after 1.5s of silence
- `MIN_UTTERANCE_MS = 500` — Reject utterances shorter than 500ms
- `MAX_UTTERANCE_MS = 30000` — Safety limit: max 30s per utterance

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

