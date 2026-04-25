import { ChatterboxModel, AutoProcessor, Tensor } from '@huggingface/transformers'
import fs from 'fs'
import path from 'path'

const VOICES_DIR = 'voices'
const SAMPLE_RATE = 24000
const TENSOR_KEYS = ['audio_features', 'audio_tokens', 'speaker_embeddings', 'speaker_features']
const TYPED_ARRAYS = { float32: Float32Array, int64: BigInt64Array }

function readWavMono(wavPath) {
  const buf = fs.readFileSync(wavPath)
  const sampleRate = buf.readUInt32LE(24)
  const channels = buf.readUInt16LE(22)
  const bitsPerSample = buf.readUInt16LE(34)
  let dataOffset = 44
  if (buf.slice(36, 40).toString('ascii') !== 'data') dataOffset = buf.indexOf('data', 12) + 8
  const dataLen = buf.readUInt32LE(dataOffset - 4)
  const numSamples = dataLen / (bitsPerSample / 8) / channels
  const mono = new Float32Array(numSamples)
  if (bitsPerSample === 16) {
    for (let i = 0; i < numSamples; i++) {
      let s = 0
      for (let c = 0; c < channels; c++) s += buf.readInt16LE(dataOffset + (i * channels + c) * 2)
      mono[i] = (s / channels) / 32768
    }
  } else if (bitsPerSample === 32) {
    for (let i = 0; i < numSamples; i++) {
      let s = 0
      for (let c = 0; c < channels; c++) s += buf.readFloatLE(dataOffset + (i * channels + c) * 4)
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

function tensorToBytes(tensor) {
  const data = tensor.data
  return Buffer.from(data.buffer, data.byteOffset, data.byteLength)
}

async function encodeOne(model, name) {
  const wavPath = path.join(VOICES_DIR, `${name}.wav`)
  const binPath = path.join(VOICES_DIR, `${name}.embedding.bin`)
  const jsonPath = path.join(VOICES_DIR, `${name}.embedding.json`)

  const wavStat = fs.statSync(wavPath)
  if (fs.existsSync(binPath) && fs.existsSync(jsonPath)) {
    const binStat = fs.statSync(binPath)
    if (binStat.mtimeMs >= wavStat.mtimeMs) {
      console.log(`[encode-speakers] ${name}: cached (.bin newer than .wav), skip`)
      return { name, skipped: true }
    }
  }

  const { audio, sampleRate } = readWavMono(wavPath)
  const resampled = resampleMono(audio, sampleRate, SAMPLE_RATE)
  console.log(`[encode-speakers] ${name}: ${audio.length} samples @ ${sampleRate}Hz -> ${resampled.length} @ ${SAMPLE_RATE}Hz`)

  const t0 = Date.now()
  const tensors = await model.encode_speech(new Tensor('float32', resampled, [1, resampled.length]))
  console.log(`[encode-speakers] ${name}: encode_speech ${Date.now() - t0}ms`)

  const manifest = { sampleRate: SAMPLE_RATE, generatedAt: new Date().toISOString(), tensors: {} }
  const chunks = []
  let offset = 0
  for (const key of TENSOR_KEYS) {
    const t = tensors[key]
    if (!t) throw new Error(`Missing tensor ${key} in encode_speech output`)
    const bytes = tensorToBytes(t)
    manifest.tensors[key] = { dtype: t.type, dims: t.dims, byteOffset: offset, byteLength: bytes.byteLength }
    chunks.push(bytes)
    offset += bytes.byteLength
  }
  fs.writeFileSync(binPath, Buffer.concat(chunks))
  fs.writeFileSync(jsonPath, JSON.stringify(manifest, null, 2))
  console.log(`[encode-speakers] ${name}: wrote ${binPath} (${offset} bytes)`)
  return { name, bytes: offset }
}

async function main() {
  if (!fs.existsSync(VOICES_DIR)) throw new Error(`${VOICES_DIR} not found`)
  const wavs = fs.readdirSync(VOICES_DIR).filter(f => f.endsWith('.wav'))
  if (!wavs.length) { console.log('[encode-speakers] no WAVs to encode'); return }
  console.log(`[encode-speakers] loading ChatterboxModel...`)
  const t0 = Date.now()
  await AutoProcessor.from_pretrained('ResembleAI/chatterbox-turbo-ONNX')
  const model = await ChatterboxModel.from_pretrained('ResembleAI/chatterbox-turbo-ONNX', {
    dtype: { embed_tokens: 'q4f16', speech_encoder: 'q4f16', language_model: 'q4f16', conditional_decoder: 'q4f16' },
  })
  console.log(`[encode-speakers] model loaded in ${Date.now() - t0}ms`)
  const results = []
  for (const wav of wavs) {
    const name = wav.replace(/\.wav$/, '')
    results.push(await encodeOne(model, name))
  }
  console.log(`[encode-speakers] done. ${results.filter(r => !r.skipped).length} encoded, ${results.filter(r => r.skipped).length} skipped`)
}

main().catch(e => { console.error('[encode-speakers] FAILED:', e); process.exit(1) })
