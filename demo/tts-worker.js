import { ChatterboxModel, AutoProcessor, Tensor, env } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.2.0'

const MODEL_ID = 'onnx-community/chatterbox-ONNX'
const SAMPLE_RATE = 24000
const TYPED_ARRAYS = { float32: Float32Array, int64: BigInt64Array }

// onnx-community/chatterbox-ONNX only ships q4f16/q4 for language_model; others are fp32
const DTYPE_WEBGPU = { embed_tokens: 'fp32', speech_encoder: 'fp32', language_model: 'q4f16', conditional_decoder: 'fp32' }
const DTYPE_WASM = { embed_tokens: 'fp32', speech_encoder: 'fp32', language_model: 'q4', conditional_decoder: 'fp32' }

let model = null, processor = null
let activeVoice = null
let activeEmbeddings = null
let aborted = false

async function checkWebGPU() {
  if (typeof navigator === 'undefined' || !navigator.gpu) return false
  try { return Boolean(await navigator.gpu.requestAdapter()) } catch { return false }
}

async function loadModel() {
  const useDevice = (await checkWebGPU()) ? 'webgpu' : 'wasm'
  self.postMessage({ type: 'status', status: `Loading model (${useDevice})…` })
  processor = await AutoProcessor.from_pretrained(MODEL_ID)
  model = await ChatterboxModel.from_pretrained(MODEL_ID, {
    device: useDevice,
    dtype: useDevice === 'webgpu' ? DTYPE_WEBGPU : DTYPE_WASM,
    progress_callback: (p) => {
      if (p.status === 'progress') {
        const pct = p.progress != null ? ` ${Math.round(p.progress)}%` : ''
        self.postMessage({ type: 'status', status: `Loading: ${p.file}${pct}` })
      }
    },
  })
  return useDevice
}

async function loadVoice(name) {
  const binResp = await fetch(`./voices/${name}.embedding.bin`)
  const jsonResp = await fetch(`./voices/${name}.embedding.json`)
  if (!binResp.ok || !jsonResp.ok) throw new Error(`voice ${name}: pre-encoded embedding not found (run tools/encode-speakers.mjs)`)
  const manifest = await jsonResp.json()
  const buf = new Uint8Array(await binResp.arrayBuffer())
  const tensors = {}
  for (const [key, meta] of Object.entries(manifest.tensors)) {
    const Ctor = TYPED_ARRAYS[meta.dtype]
    if (!Ctor) throw new Error(`unsupported dtype ${meta.dtype}`)
    const slice = buf.buffer.slice(buf.byteOffset + meta.byteOffset, buf.byteOffset + meta.byteOffset + meta.byteLength)
    tensors[key] = new Tensor(meta.dtype, new Ctor(slice), meta.dims)
  }
  activeVoice = name
  activeEmbeddings = tensors
}

async function generate(text) {
  if (!model || !processor) throw new Error('Model not loaded')
  if (!activeEmbeddings) throw new Error('Voice not loaded')
  aborted = false
  const inputs = await processor._call(text)
  const waveform = await model.generate({ ...inputs, ...activeEmbeddings, exaggeration: 0.5, max_new_tokens: 256 })
  if (aborted) return
  const data = waveform.data
  const buf = data.buffer.slice(data.byteOffset, data.byteOffset + data.byteLength)
  self.postMessage({ type: 'audio_chunk', data: buf }, [buf])
  self.postMessage({ type: 'stream_ended' })
}

self.onmessage = async (e) => {
  const { type } = e.data
  try {
    if (type === 'load') {
      const device = await loadModel()
      self.postMessage({ type: 'status', status: `Model loaded (${device})` })
      const manifest = await fetch('./voices/manifest.json').then((r) => r.json())
      const voices = manifest.map((f) => f.replace(/\.wav$/, ''))
      self.postMessage({ type: 'voices_loaded', voices, defaultVoice: voices[0] || 'cleetus' })
    } else if (type === 'load_voice') {
      const name = e.data.voice
      self.postMessage({ type: 'status', status: `Loading voice: ${name}` })
      await loadVoice(name)
      self.postMessage({ type: 'loaded' })
    } else if (type === 'generate') {
      await generate(e.data.data?.text ?? e.data.text)
    } else if (type === 'cancel') {
      aborted = true
    }
  } catch (err) {
    self.postMessage({ type: 'error', error: err.message })
  }
}
