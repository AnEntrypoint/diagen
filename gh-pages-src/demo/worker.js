import { AutoModelForCausalLM, AutoProcessor, TextStreamer, env } from './transformers.min.js?v=62'

const MODEL_BASE = './model'
const CHUNKS = {
  'model_q4f16_q4f16.onnx': {
    stem: 'model_q4f16',
    sizes: [103809024, 103809024, 103809024, 103809024, 67767486]
  },
}

self.addEventListener('unhandledrejection', (e) => {
  const msg = e.reason?.message || (typeof e.reason === 'string' ? e.reason : null) || JSON.stringify(e.reason) || String(e)
  self.postMessage({ type: 'error', message: msg })
})

function fetchChunked(stem, sizes) {
  const total = sizes.reduce((s, n) => s + n, 0)
  let idx = 0, off = 0
  const stream = new ReadableStream({
    async pull(controller) {
      if (idx >= sizes.length) {
        self.postMessage({ type: 'progress', progress: { progress: 82, file: stem } })
        controller.close(); return
      }
      const i = idx++
      self.postMessage({ type: 'progress', progress: { progress: Math.round(off / total * 80), file: `${stem}.part${i}` } })
      let buf = await origFetch(`${MODEL_BASE}/onnx/${stem}.onnx.part${i}`).then(r => r.arrayBuffer())
      off += buf.byteLength
      controller.enqueue(new Uint8Array(buf))
      buf = null
    }
  })
  return new Response(stream, { status: 200, headers: { 'Content-Type': 'application/octet-stream', 'Content-Length': String(total) } })
}

const origFetch = self.fetch.bind(self)
self.fetch = async (input, init) => {
  const url = typeof input === 'string' ? input : input.url
  for (const [fname, { stem, sizes }] of Object.entries(CHUNKS)) {
    if (url.endsWith(fname)) return fetchChunked(stem, sizes)
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
env.backends.onnx.wasm.numThreads = 4

// Bust stale transformers-cache entries: JSON configs always, plus any ONNX that was cached
// before chunked reassembly was in place (those would be the small 780KB stub file, not the
// full 453MB self-contained ONNX we serve via the fetch interceptor).
const DECODER_ONNX_MIN_SIZE = 338102507
const cacheBust = (async () => {
  try {
    const c = await caches.open('transformers-cache')
    const keys = await c.keys()
    for (const k of keys) {
      if (k.url.includes('/model/') && k.url.endsWith('.json')) { await c.delete(k); continue }
      if (k.url.includes('model_q4f16_q4f16.onnx') || k.url.includes('model_q4f16_quantized.onnx') || k.url.includes('model_q4f16.onnx')) {
        const resp = await c.match(k)
        if (resp) {
          const buf = await resp.clone().arrayBuffer()
          if (buf.byteLength < DECODER_ONNX_MIN_SIZE || new Uint8Array(buf.slice(0,1))[0] === 0x3C) {
            await c.delete(k)
            console.log('[worker] Cleared stale ONNX from cache:', k.url, buf.byteLength)
          }
        }
      }
    }
  } catch(e) {}
})()

const MODEL_ID = 'model'
const DTYPE = 'q4f16'

let model = null, processor = null
let loading = false, loadError = null, activeDevice = 'wasm'

async function tryLoadModel(device, progress) {
  return AutoModelForCausalLM.from_pretrained(MODEL_ID, {
    dtype: DTYPE, device, progress_callback: progress,
    model_file_name: 'model_q4f16'
  })
}

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
      model = await tryLoadModel('wasm', progress)
      activeDevice = 'wasm'
      loading = false
      self.postMessage({ type: 'loaded', id, device: activeDevice })
    } catch (err) {
      loading = false; loadError = err.message
      self.postMessage({ type: 'error', message: err.message, id })
    }
    return
  }


  if (type === 'reset') {
    self.postMessage({ type: 'reset', id })
    return
  }
  if (type === 'generate') {
    if (!model) { self.postMessage({ type: 'error', message: loadError ?? 'Model not loaded', id }); return }
    const { messages, config = {} } = e.data
    try {
      const tokens = []
      const formatted = messages.map(m => ({ role: m.role, content: m.content }))
      const promptText = processor.apply_chat_template(formatted, { add_generation_prompt: config.addGenerationPrompt !== false, tokenize: false })
      const { input_ids, attention_mask } = await processor.tokenizer(promptText, { return_tensors: 'pt' })
      const streamer = new TextStreamer(processor.tokenizer, {
        skip_prompt: true, skip_special_tokens: true,
        callback_function: (token) => { tokens.push(token); self.postMessage({ type: 'token', token, id }) },
      })
      const genArgs = { max_new_tokens: config.maxNewTokens ?? 300, temperature: config.temperature ?? 0.7, repetition_penalty: config.repetitionPenalty ?? 1.0, do_sample: true, streamer }
      const result = await model.generate({ input_ids, attention_mask, ...genArgs })
      self.postMessage({ type: 'result', text: tokens.join(''), id })
    } catch (err) {
      self.postMessage({ type: 'error', message: err.message, id })
    }
    return
  }
}
