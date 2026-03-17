import { pipeline, TextStreamer } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3/dist/transformers.min.js'

const LLM_MODEL = 'onnx-community/Qwen2.5-0.5B-Instruct'
const TTS_MODEL = 'onnx-community/Kokoro-82M-v1.0-ONNX'
const LLM_DTYPE = 'q4'

let generator = null
let synthesizer = null
let llmLoading = false
let ttsLoading = false
let loadError = null

self.onmessage = async (e) => {
  const { type, id } = e.data

  if (type === 'load') {
    if (generator && synthesizer) { self.postMessage({ type: 'loaded', id }); return }
    if (llmLoading) return
    llmLoading = true
    ttsLoading = true
    try {
      generator = await pipeline('text-generation', LLM_MODEL, {
        dtype: LLM_DTYPE,
        device: 'webgpu',
        progress_callback: (p) => self.postMessage({ type: 'progress', progress: p }),
      }).catch(() => pipeline('text-generation', LLM_MODEL, {
        dtype: LLM_DTYPE,
        device: 'wasm',
        progress_callback: (p) => self.postMessage({ type: 'progress', progress: p }),
      }))
      llmLoading = false

      try {
        synthesizer = await pipeline('text-to-speech', TTS_MODEL, {
          dtype: 'fp32',
          device: 'wasm',
          progress_callback: (p) => self.postMessage({ type: 'tts_progress', progress: p }),
        })
      } catch (ttsErr) {
        console.warn('[worker] TTS model failed to load:', ttsErr.message)
      }
      ttsLoading = false

      self.postMessage({ type: 'loaded', ttsLoaded: !!synthesizer, id })
    } catch (err) {
      llmLoading = false
      ttsLoading = false
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

  if (type === 'synthesize') {
    if (!synthesizer) {
      self.postMessage({ type: 'error', message: 'TTS model not loaded', id })
      return
    }
    const { text } = e.data
    try {
      const out = await synthesizer(text, { voice: 'af_heart' })
      const { audio, sampling_rate } = out
      self.postMessage({ type: 'audio', audio, sampling_rate, id }, [audio.buffer])
    } catch (err) {
      self.postMessage({ type: 'error', message: err.message, id })
    }
    return
  }
}
