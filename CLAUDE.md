# Diagen Project Notes

## Model — Non-Negotiable

The browser demo uses Qwen3.5-VL 0.8B abliterated with identity-override LoRA merged, quantized to q4 ONNX.

- Source: bobber/routangseng-qwen35-0.8b-abliterated-lora-onnx (HuggingFace)
- Quantization: q4 via MatMulNBitsQuantizer (decoder), fp16 (embed_tokens, WebGPU), q8 (vision_encoder)
- Delivery: real git blobs split into ≤99MB chunks served from GitHub Pages
- No LFS. No remote model fetching. Local files only in browser worker.

This model choice is non-negotiable. Do not substitute with a different model.

## Dependencies

- `sttttsmodels` is a local file dep (`file:../sttttsmodels`). Clone it as a sibling directory or `npm install` will fail.
- `audio2afan` runs a postinstall download script. Use `npm install --ignore-scripts` when offline, then download models separately via `npm run download-models`.

## Testing

Run `npm test` (vitest). 120 tests across 7 files. Pure functions are extracted into shared modules (`server-utils.mjs`, `animation-core.mjs`, `tokenizer.mjs`) that both source files and tests import.

## Quantize Workflow

To switch to Qwen2.5-0.5B abliterated (faster, roleplay-focused):
- Run `.github/workflows/quantize-model.yml` via GitHub Actions (workflow_dispatch)
- Requires `HF_TOKEN` secret if downloading gated models
- Workflow: downloads PyTorch weights → optimum-cli ONNX export → MatMulNBitsQuantizer q4f16 → splits into ≤99MB parts → commits model files + updates worker.js to main
- `*.py` is in .gitignore — Python helpers are inlined as workflow heredocs, not separate files

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
