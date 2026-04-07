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
