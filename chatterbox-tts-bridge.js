import { ChatterboxModel, AutoProcessor, Tensor } from '@huggingface/transformers'
import fs from 'fs'
import path from 'path'

const SAMPLE_RATE = 24000
const MAX_CHUNK_CHARS = 200

const ABBREVIATIONS = new Set([
  'mr','mrs','ms','dr','prof','sr','jr','st','ave','blvd',
  'gen','gov','sgt','cpl','pvt','capt','lt','col','maj',
  'etc','vs','vol','dept','est','approx','inc','ltd','co',
])

function splitSentences(text) {
  const chunks = []
  let buf = ''
  for (let i = 0; i < text.length; i++) {
    buf += text[i]
    if ('.!?'.includes(text[i]) && text[i + 1] === ' ') {
      const word = buf.split(/\s+/).at(-2)?.replace(/[^a-z]/gi, '').toLowerCase()
      if (!ABBREVIATIONS.has(word)) { chunks.push(buf.trim()); buf = '' }
    }
  }
  if (buf.trim()) chunks.push(buf.trim())
  return chunks.length ? chunks : [text]
}

function splitTextIntoChunks(text) {
  const trimmed = text.trim()
  if (!trimmed) return []
  if (trimmed.length <= MAX_CHUNK_CHARS) return [trimmed]
  const sentences = splitSentences(trimmed)
  const chunks = []
  let cur = ''
  for (const s of sentences) {
    if (cur && (cur + ' ' + s).length > MAX_CHUNK_CHARS) { chunks.push(cur); cur = s }
    else cur = cur ? cur + ' ' + s : s
  }
  if (cur) chunks.push(cur)
  return chunks
}

function readWavMono(wavPath) {
  const buf = fs.readFileSync(wavPath)
  const sampleRate = buf.readUInt32LE(24)
  const channels = buf.readUInt16LE(22)
  const bitsPerSample = buf.readUInt16LE(34)
  let dataOffset = 44
  if (buf.slice(36, 40).toString('ascii') !== 'data') {
    dataOffset = buf.indexOf('data', 12) + 8
  }
  const dataLen = buf.readUInt32LE(dataOffset - 4)
  const numSamples = dataLen / (bitsPerSample / 8) / channels
  const mono = new Float32Array(numSamples)
  if (bitsPerSample === 16) {
    for (let i = 0; i < numSamples; i++) {
      let s = 0
      for (let c = 0; c < channels; c++) {
        s += buf.readInt16LE(dataOffset + (i * channels + c) * 2)
      }
      mono[i] = (s / channels) / 32768
    }
  } else if (bitsPerSample === 32) {
    for (let i = 0; i < numSamples; i++) {
      let s = 0
      for (let c = 0; c < channels; c++) {
        s += buf.readFloatLE(dataOffset + (i * channels + c) * 4)
      }
      mono[i] = s / channels
    }
  }
  return { audio: mono, sampleRate }
}

function resampleMono(audio, fromRate, toRate) {
  if (fromRate === toRate) return audio
  const ratio = fromRate / toRate
  const newLen = Math.round(audio.length / ratio)
  const out = new Float32Array(newLen)
  for (let i = 0; i < newLen; i++) {
    const idx = i * ratio
    const lo = Math.floor(idx)
    const hi = Math.min(lo + 1, audio.length - 1)
    out[i] = audio[lo] * (1 - (idx - lo)) + audio[hi] * (idx - lo)
  }
  return out
}

let model = null
let processor = null
let speakerEmbeddings = null
let loadPromise = null

async function ensureModel() {
  if (model && processor) return
  if (loadPromise) return loadPromise
  loadPromise = (async () => {
    console.log('[chatterbox] loading ChatterboxModel (ResembleAI/chatterbox-turbo-ONNX)...')
    processor = await AutoProcessor.from_pretrained('ResembleAI/chatterbox-turbo-ONNX')
    model = await ChatterboxModel.from_pretrained('ResembleAI/chatterbox-turbo-ONNX', {
      dtype: { embed_tokens: 'q4f16', speech_encoder: 'q4f16', language_model: 'q4f16', conditional_decoder: 'q4f16' },
    })
    console.log('[chatterbox] model loaded')
  })()
  return loadPromise
}

export async function setRefVoice(wavPath) {
  await ensureModel()
  const { audio, sampleRate } = readWavMono(wavPath)
  const resampled = resampleMono(audio, sampleRate, SAMPLE_RATE)
  const tensor = new Tensor('float32', resampled, [1, resampled.length])
  speakerEmbeddings = await model.encode_speech(tensor)
  console.log(`[chatterbox] speaker encoded from ${path.basename(wavPath)}`)
}

async function generateChunk(text, signal) {
  if (signal?.aborted) return null
  const inputs = await processor._call(text)
  const waveform = await model.generate({ ...inputs, ...speakerEmbeddings, exaggeration: 0.5, max_new_tokens: 256 })
  if (signal?.aborted) return null
  return new Float32Array(waveform.data.buffer.slice(waveform.data.byteOffset, waveform.data.byteOffset + waveform.data.byteLength))
}

export async function synthesize(text, _refPath, _refText, signal) {
  if (!text) throw new Error('text required')
  await ensureModel()
  if (!speakerEmbeddings) throw new Error('call setRefVoice() before synthesize()')
  if (signal?.aborted) return null
  const chunks = splitTextIntoChunks(text)
  const parts = []
  for (const chunk of chunks) {
    if (signal?.aborted) break
    const audio = await generateChunk(chunk, signal)
    if (audio) parts.push(audio)
  }
  if (!parts.length) return null
  const total = parts.reduce((s, a) => s + a.length, 0)
  const out = new Float32Array(total)
  let off = 0
  for (const p of parts) { out.set(p, off); off += p.length }
  return { audio: out, sampleRate: SAMPLE_RATE }
}

export async function synthesizeStream(text, _refPath, _refText, onChunk, signal) {
  if (!text) throw new Error('text required')
  if (typeof onChunk !== 'function') throw new Error('onChunk required')
  await ensureModel()
  if (!speakerEmbeddings) throw new Error('call setRefVoice() before synthesizeStream()')
  const chunks = splitTextIntoChunks(text)
  for (const chunk of chunks) {
    if (signal?.aborted) break
    const audio = await generateChunk(chunk, signal)
    if (audio && !signal?.aborted) onChunk(audio, SAMPLE_RATE)
  }
  return { sampleRate: SAMPLE_RATE }
}

export function getDebugState() {
  return {
    modelLoaded: Boolean(model && processor),
    speakerEncoded: Boolean(speakerEmbeddings),
    loading: Boolean(loadPromise && !(model && processor)),
  }
}
