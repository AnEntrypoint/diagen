import { AutoProcessor, Qwen3_5ForConditionalGeneration, TextStreamer, env } from './transformers.min.js?v=40'

const MODEL_BASE = './model'
const VISION_ENCODER_STUB = 'CAg6fQooCgxwaXhlbF92YWx1ZXMSDmltYWdlX2ZlYXR1cmVzIghJZGVudGl0eRITdmlzaW9uX2VuY29kZXJfc3R1YlocCgxwaXhlbF92YWx1ZXMSDAoKCAESBgoACgAKAGIeCg5pbWFnZV9mZWF0dXJlcxIMCgoIARIGCgAKAAoAQgQKABAR'
const CHUNKS = {
  'decoder_model_merged_weights.bin': {
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
      let buf = await origFetch(`${MODEL_BASE}/onnx/${stem}.part${i}`).then(r => r.arrayBuffer())
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
  if (url.endsWith('vision_encoder_quantized.onnx')) {
    const bin = Uint8Array.from(atob(VISION_ENCODER_STUB), c => c.charCodeAt(0))
    return new Response(bin.buffer, { status: 200, headers: { 'Content-Type': 'application/octet-stream' } })
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

// Bust stale transformers-cache entries for JSON config files only (ONNX files are immutable)
const cacheBust = (async () => {
  try {
    const c = await caches.open('transformers-cache')
    const keys = await c.keys()
    for (const k of keys) {
      if (k.url.includes('/model/') && k.url.endsWith('.json')) await c.delete(k)
    }
  } catch(e) {}
})()

const MODEL_ID = 'model'
const DTYPE = { embed_tokens: 'q8', vision_encoder: 'q8', decoder_model_merged: 'q4f16' }

let model = null, processor = null
let loading = false, loadError = null, activeDevice = 'wasm'
// KV cache: { inputIds: BigInt64Array, pastKeyValues: object }
let kvcache = null

async function tryLoadModel(device, progress) {
  return Qwen3_5ForConditionalGeneration.from_pretrained(MODEL_ID, {
    dtype: DTYPE, device, progress_callback: progress,
    model_file_name: 'decoder_model_merged'
  })
}

function cacheKey(ids) { return ids.join(',') }

function findCachePrefix(inputIds) {
  if (!kvcache) return null
  const cached = kvcache.inputIds
  if (cached.length >= inputIds.length) return null
  for (let i = 0; i < cached.length; i++) {
    if (cached[i] !== inputIds[i]) return null
  }
  return kvcache
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
      try {
        model = await tryLoadModel('webgpu', progress)
        activeDevice = 'webgpu'
      } catch (gpuErr) {
        self.postMessage({ type: 'progress', progress: { progress: 0, file: `WebGPU failed (${gpuErr.message.slice(0,60)}), falling back to WASM…` } })
        model = await tryLoadModel('wasm', progress)
        activeDevice = 'wasm'
      }
      loading = false
      self.postMessage({ type: 'loaded', id, device: activeDevice })
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
      const { input_ids, attention_mask } = await processor.tokenizer(promptText, { return_tensors: 'pt' })
      const inputArr = Array.from(input_ids.data)
      const hit = findCachePrefix(inputArr)
      const streamer = new TextStreamer(processor.tokenizer, {
        skip_prompt: true, skip_special_tokens: true,
        callback_function: (token) => { tokens.push(token); self.postMessage({ type: 'token', token, id }) },
      })
      const genArgs = { max_new_tokens: config.maxNewTokens ?? 300, temperature: config.temperature ?? 0.7, repetition_penalty: config.repetitionPenalty ?? 1.0, do_sample: true, streamer, return_dict_in_generate: true }
      let result
      if (hit) {
        const newIds = input_ids.slice(null, [hit.inputIds.length, null])
        const newMask = attention_mask.slice(null, [hit.inputIds.length, null])
        result = await model.generate({ input_ids: newIds, attention_mask: newMask, past_key_values: hit.pastKeyValues, ...genArgs })
      } else {
        result = await model.generate({ input_ids, attention_mask, ...genArgs })
      }
      // Cache the full prompt KV state for next turn reuse
      if (result.past_key_values) {
        kvcache = { inputIds: inputArr, pastKeyValues: result.past_key_values }
      }
      self.postMessage({ type: 'result', text: tokens.join(''), id })
    } catch (err) {
      kvcache = null
      self.postMessage({ type: 'error', message: err.message, id })
    }
    return
  }
}
