import { pipeline, TextStreamer } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3/dist/transformers.min.js'

const MODEL_ID = 'onnx-community/Qwen2.5-0.5B-Instruct'
const DTYPE = 'q4'

let generator = null
let loading = false
let loadError = null

self.onmessage = async (e) => {
  const { type, id } = e.data

  if (type === 'load') {
    if (generator) { self.postMessage({ type: 'loaded', id }); return }
    if (loading) return
    loading = true
    try {
      generator = await pipeline('text-generation', MODEL_ID, {
        dtype: DTYPE,
        device: 'webgpu',
        progress_callback: (p) => self.postMessage({ type: 'progress', progress: p }),
      }).catch(async () => {
        return pipeline('text-generation', MODEL_ID, {
          dtype: DTYPE,
          device: 'wasm',
          progress_callback: (p) => self.postMessage({ type: 'progress', progress: p }),
        })
      })
      loading = false
      self.postMessage({ type: 'loaded', id })
    } catch (err) {
      loading = false
      loadError = err.message
      self.postMessage({ type: 'error', message: err.message, id })
    }
    return
  }

  if (type === 'generate') {
    if (!generator) {
      self.postMessage({ type: 'error', message: loadError ?? 'Model not loaded', id })
      return
    }
    const { messages, config = {} } = e.data
    try {
      const tokens = []
      const streamer = new TextStreamer(generator.tokenizer, {
        skip_prompt: true,
        skip_special_tokens: true,
        callback_function: (token) => {
          tokens.push(token)
          self.postMessage({ type: 'token', token, id })
        },
      })
      await generator(messages, {
        max_new_tokens: config.maxNewTokens ?? 300,
        temperature: config.temperature ?? 0.7,
        do_sample: config.temperature > 0,
        streamer,
      })
      self.postMessage({ type: 'result', text: tokens.join(''), id })
    } catch (err) {
      self.postMessage({ type: 'error', message: err.message, id })
    }
    return
  }
}
