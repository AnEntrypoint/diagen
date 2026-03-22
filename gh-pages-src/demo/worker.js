import { AutoProcessor, Qwen3_5ForConditionalGeneration, TextStreamer, env } from './transformers.min.js?v=32'

const MODEL_BASE = './model'
const CHUNKS = {
  'decoder_model_merged_q4f16.onnx': {
    stem: 'decoder_model_merged_q4f16.onnx',
    sizes: [103809024, 103809024, 103809024, 103809024, 59248715]
  },
  'embed_tokens_quantized.onnx': {
    stem: 'embed_tokens_q8',
    sizes: [103809024, 103809024, 46662328]
  },
}

self.addEventListener('unhandledrejection', (e) => {
  const msg = e.reason?.message || (typeof e.reason === 'string' ? e.reason : null) || JSON.stringify(e.reason) || String(e)
  self.postMessage({ type: 'error', message: msg })
})

async function fetchChunked(stem, sizes) {
  const total = sizes.reduce((s, n) => s + n, 0)
  const out = new Uint8Array(total)
  let off = 0
  for (let i = 0; i < sizes.length; i++) {
    self.postMessage({ type: 'progress', progress: { progress: Math.round(off / total * 80), file: `${stem}.part${i}` } })
    let buf = await origFetch(`${MODEL_BASE}/onnx/${stem}.part${i}`).then(r => r.arrayBuffer())
    out.set(new Uint8Array(buf), off)
    off += buf.byteLength
    buf = null
  }
  self.postMessage({ type: 'progress', progress: { progress: 82, file: stem } })
  return out.buffer
}

const origFetch = self.fetch.bind(self)
self.fetch = async (input, init) => {
  const url = typeof input === 'string' ? input : input.url
  for (const [fname, { stem, sizes }] of Object.entries(CHUNKS)) {
    if (url.endsWith(fname)) {
      const buf = await fetchChunked(stem, sizes)
      return new Response(buf, { status: 200, headers: { 'Content-Type': 'application/octet-stream' } })
    }
  }
  if (url.endsWith('/model/config.json')) {
    const resp = await origFetch(input, init)
    const json = await resp.json()
    json['transformers.js_config'] = {}
    return new Response(JSON.stringify(json), { status: 200, headers: { 'Content-Type': 'application/json' } })
  }
  return origFetch(input, init)
}

env.allowLocalModels = true
env.allowRemoteModels = false
env.localModelPath = './'
env.fetch = self.fetch
env.backends.onnx.wasm.numThreads = 1

// Bust stale transformers-cache entries for ALL model files
const cacheBust = (async () => {
  try {
    const c = await caches.open('transformers-cache')
    const keys = await c.keys()
    for (const k of keys) {
      if (k.url.includes('/model/')) await c.delete(k)
    }
  } catch(e) {}
})()

const MODEL_ID = 'model'
const DTYPE = { embed_tokens: 'q8', vision_encoder: 'q8', decoder_model_merged: 'q4f16' }

let model = null, processor = null
let loading = false, loadError = null

self.onmessage = async (e) => {
  const { type, id } = e.data

  if (type === 'load') {
    if (model) { self.postMessage({ type: 'loaded', id }); return }
    if (loading) return
    loading = true
    await cacheBust
    try {
      const progress = (p) => self.postMessage({ type: 'progress', progress: p })
      processor = await AutoProcessor.from_pretrained(MODEL_ID, { progress_callback: progress })
      model = await Qwen3_5ForConditionalGeneration.from_pretrained(MODEL_ID, {
        dtype: DTYPE, device: 'wasm', progress_callback: progress,
        model_file_name: 'decoder_model_merged'
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
      const dbg = { modelType: model.config?.model_type, sessions: Object.keys(model.sessions || {}), hasFwd: typeof model._forward }
      self.postMessage({ type: 'debug', dbg, id })
      const tokens = []
      const formatted = messages.map(m => ({ role: m.role, content: [{ type: 'text', text: m.content }] }))
      const promptText = processor.apply_chat_template(formatted, { add_generation_prompt: config.addGenerationPrompt !== false, tokenize: false })
      const inputs = await processor.tokenizer(promptText, { return_tensors: 'pt' })
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
