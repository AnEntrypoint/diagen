import { ChatterboxModel, AutoProcessor, Tensor, env } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.2.0/+esm'

env.localModelPath = './model/chatterbox/'
env.allowRemoteModels = false

const MODEL_ID = 'ResembleAI/chatterbox-turbo-ONNX'
const MODEL_BASE = env.localModelPath + MODEL_ID + '/'
const SAMPLE_RATE = 24000
let chunkManifest = null

const _origFetch = self.fetch.bind(self)

async function getChunkManifest() {
  if (chunkManifest) return chunkManifest
  const resp = await _origFetch(MODEL_BASE + 'chunks.json')
  if (!resp.ok) throw new Error('chunks.json fetch failed: ' + resp.status)
  chunkManifest = await resp.json()
  return chunkManifest
}

self.fetch = async (input, init) => {
  const url = typeof input === 'string' ? input : input.url
  const manifest = await getChunkManifest().catch(() => [])
  const entry = manifest.find(e => url.endsWith('/' + e.file) || url.endsWith(e.file.replace(/\//g, '/')))
  if (!entry) return _origFetch(input, init)
  const parts = await Promise.all(
    Array.from({ length: entry.parts }, (_, i) => _origFetch(url + '.part' + i).then(r => {
      if (!r.ok) throw new Error('chunk fetch failed: ' + url + '.part' + i + ' ' + r.status)
      return r.arrayBuffer()
    }))
  )
  const total = parts.reduce((s, b) => s + b.byteLength, 0)
  const merged = new Uint8Array(total)
  let off = 0
  for (const b of parts) { merged.set(new Uint8Array(b), off); off += b.byteLength }
  return new Response(merged.buffer, { status: 200, headers: { 'Content-Type': 'application/octet-stream' } })
}

let model = null
let processor = null
let speakerEmbeddings = null
let aborted = false

const DTYPE_CONFIGS = {
  wasm:   { embed_tokens: 'q4', speech_encoder: 'q4', language_model: 'q4', conditional_decoder: 'q4' },
  webgpu: { embed_tokens: 'q4f16', speech_encoder: 'q4f16', language_model: 'q4f16', conditional_decoder: 'q4f16' },
}

async function detectDevice() {
  if (typeof navigator === 'undefined' || !navigator.gpu) return 'wasm'
  try {
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' })
    return adapter ? 'webgpu' : 'wasm'
  } catch { return 'wasm' }
}

async function load() {
  self.postMessage({ type: 'status', status: 'Loading Chatterbox Turbo model…' })
  const device = await detectDevice()
  const dtype = DTYPE_CONFIGS[device] ?? DTYPE_CONFIGS.wasm
  processor = await AutoProcessor.from_pretrained(MODEL_ID, {
    progress_callback: (p) => {
      if (p.status === 'progress') {
        const pct = p.progress != null ? ` ${Math.round(p.progress)}%` : ''
        self.postMessage({ type: 'status', status: `Loading: ${p.file}${pct}` })
      }
    },
  })
  model = await ChatterboxModel.from_pretrained(MODEL_ID, {
    device,
    dtype,
    progress_callback: (p) => {
      if (p.status === 'progress') {
        const pct = p.progress != null ? ` ${Math.round(p.progress)}%` : ''
        self.postMessage({ type: 'status', status: `Loading: ${p.file}${pct}` })
      }
    },
  })
  self.postMessage({ type: 'status', status: `Model loaded (${device})` })
}

async function loadVoice(voiceName) {
  self.postMessage({ type: 'status', status: `Encoding speaker: ${voiceName}` })
  const resp = await fetch(`./voices/${voiceName}.wav`)
  if (!resp.ok) throw new Error(`voice fetch failed: ${voiceName}.wav`)
  const arrayBuf = await resp.arrayBuffer()
  const audioCtx = new OfflineAudioContext(1, 1, SAMPLE_RATE)
  const decoded = await audioCtx.decodeAudioData(arrayBuf)
  const mono = decoded.getChannelData(0)
  const tensor = new Tensor('float32', mono, [1, mono.length])
  speakerEmbeddings = await model.encode_speech(tensor)
  self.postMessage({ type: 'loaded' })
}

const ABBREVIATIONS = new Set([
  'mr','mrs','ms','dr','prof','sr','jr','st','ave','blvd',
  'gen','gov','sgt','cpl','pvt','capt','lt','col','maj',
  'etc','vs','vol','dept','est','approx','inc','ltd','co',
])

function splitText(text) {
  const trimmed = text.trim()
  if (!trimmed) return []
  if (trimmed.length <= 200) return [trimmed]
  const chunks = []
  let buf = ''
  for (let i = 0; i < trimmed.length; i++) {
    buf += trimmed[i]
    if ('.!?'.includes(trimmed[i]) && trimmed[i + 1] === ' ') {
      const word = buf.split(/\s+/).at(-2)?.replace(/[^a-z]/gi, '').toLowerCase()
      if (!ABBREVIATIONS.has(word)) { chunks.push(buf.trim()); buf = '' }
    }
  }
  if (buf.trim()) chunks.push(buf.trim())
  return chunks.length ? chunks : [trimmed]
}

async function generate(text) {
  if (!model || !processor) throw new Error('Model not loaded')
  if (!speakerEmbeddings) throw new Error('No speaker encoded — load a voice first')
  aborted = false
  for (const chunk of splitText(text)) {
    if (aborted) break
    const inputs = await processor._call(chunk)
    const waveform = await model.generate({ ...inputs, ...speakerEmbeddings, exaggeration: 0.5, max_new_tokens: 256 })
    if (aborted) break
    const buf = waveform.data.buffer.slice(waveform.data.byteOffset, waveform.data.byteOffset + waveform.data.byteLength)
    self.postMessage({ type: 'audio_chunk', data: buf }, [buf])
  }
  if (!aborted) self.postMessage({ type: 'stream_ended' })
}

self.onmessage = async (e) => {
  const { type, data } = e.data
  try {
    if (type === 'load') {
      await load()
      const resp = await fetch('./voices/manifest.json')
      const manifest = await resp.json()
      const voices = manifest.map(f => f.replace('.wav', ''))
      const defaultVoice = voices[0] || 'cleetus'
      self.postMessage({ type: 'voices_loaded', voices, defaultVoice })
    } else if (type === 'load_voice') {
      await loadVoice(data?.voice ?? e.data.voice)
    } else if (type === 'generate') {
      await generate(data?.text ?? e.data.data?.text)
    } else if (type === 'cancel') {
      aborted = true
    }
  } catch (err) {
    self.postMessage({ type: 'error', error: err.message })
  }
}
