# Diagen Project Notes

## Model — Non-Negotiable

The browser demo uses Qwen3.5-VL 0.8B abliterated with identity-override LoRA merged, quantized to q4 ONNX.

- Source: bobber/routangseng-qwen35-0.8b-abliterated-lora-onnx (HuggingFace)
- Quantization: q4 via MatMulNBitsQuantizer (decoder), q8 dynamic (embed_tokens), q8 (vision_encoder)
- Delivery: real git blobs split into ≤99MB chunks served from GitHub Pages
- No LFS. No remote model fetching. Local files only in browser worker.

This model choice is non-negotiable. Do not substitute with a different model.
