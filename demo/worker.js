import { AutoProcessor, Qwen3_5ForConditionalGeneration, TextStreamer, env } from './transformers.min.js?v=14'

const MODEL_BASE = './model'
const CHUNKS = {
  'decoder_model_merged_q4f16.onnx_data': { stem: 'decoder_model_merged_q4f16.onnx_data', parts: 5 },
  'embed_tokens_quantized.onnx': { stem: 'embed_tokens_q8', parts: 3 },
}

self.addEventListener('unhandledrejection', (e) => {
  self.postMessage({ type: 'error', message: String(e.reason?.message || e.reason || e) })
})

async function fetchChunked(stem, parts) {
  const buffers = await Promise.all(
    Array.from({ length: parts }, (_, i) =>
      fetch(`${MODEL_BASE}/onnx/${stem}.part${i}`).then(r => r.arrayBuffer())
    )
  )
  const total = buffers.reduce((s, b) => s + b.byteLength, 0)
  const out = new Uint8Array(total)
  let off = 0
  for (const b of buffers) { out.set(new Uint8Array(b), off); off += b.byteLength }
  return out.buffer
}

const origFetch = self.fetch.bind(self)
self.fetch = async (input, init) => {
  const url = typeof input === 'string' ? input : input.url
  for (const [fname, { stem, parts }] of Object.entries(CHUNKS)) {
    if (url.endsWith(fname)) {
      const buf = await fetchChunked(stem, parts)
      return new Response(buf, { status: 200, headers: { 'Content-Type': 'application/octet-stream' } })
    }
  }
  return origFetch(input, init)
}

env.allowLocalModels = true
env.allowRemoteModels = false
env.localModelPath = './'
env.fetch = self.fetch

const MODEL_ID = 'model'
const DTYPE = { embed_tokens: 'q8', decoder_model_merged: 'q4f16' }

let model = null, processor = null
let loading = false, loadError = null

self.onmessage = async (e) => {
  const { type, id } = e.data

  if (type === 'load') {
    if (model) { self.postMessage({ type: 'loaded', id }); return }
    if (loading) return
    loading = true
    try {
      const progress = (p) => self.postMessage({ type: 'progress', progress: p })
      processor = await AutoProcessor.from_pretrained(MODEL_ID, { progress_callback: progress })
      model = await Qwen3_5ForConditionalGeneration.from_pretrained(MODEL_ID, {
        dtype: DTYPE, device: 'wasm', progress_callback: progress
      })
      loading = false
      self.postMessage({ type: 'loaded', id })
    } catch (err) {
      loading = false; loadError = err.message
      self.postMessage({ type: 'error', message: err.message, id })
    }
    return
  }

  if (type === 'generate') {
    if (!model) { self.postMessage({ type: 'error', message: loadError ?? 'Model not loaded', id }); return }
    const { messages, config = {} } = e.data
    try {
      const tokens = []
      const formatted = messages.map(m => ({ role: m.role, content: [{ type: 'text', text: m.content }] }))
      const promptText = processor.apply_chat_template(formatted, { add_generation_prompt: config.addGenerationPrompt !== false, tokenize: false })
      const inputs = await processor(promptText)
      const streamer = new TextStreamer(processor.tokenizer, {
        skip_prompt: true, skip_special_tokens: true,
        callback_function: (token) => { tokens.push(token); self.postMessage({ type: 'token', token, id }) },
      })
      await model.generate({ ...inputs, max_new_tokens: config.maxNewTokens ?? 300, temperature: config.temperature ?? 0.7, repetition_penalty: config.repetitionPenalty ?? 1.0, do_sample: true, streamer })
      self.postMessage({ type: 'result', text: tokens.join(''), id })
    } catch (err) {
      self.postMessage({ type: 'error', message: err.message, id })
    }
    return
  }
}
