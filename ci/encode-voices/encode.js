import ort from 'onnxruntime-node'
import { readFileSync, readdirSync, writeFileSync, statSync } from 'fs'
import { join } from 'path'

const MODEL_DIR = join(process.cwd(), 'models', 'tts')
const VOICES_DIR = join(process.cwd(), 'voices')
const OUT_DIR = join(process.cwd(), 'gh-pages-src', 'demo', 'voices')

const FLOW_STATE_SHAPES = [
  { shape: [2, 1, 1000, 16, 64], dtype: 'float32' },
  { shape: [0], dtype: 'float32' },
  { shape: [1], dtype: 'int64' },
]

function initFlowState(session) {
  const state = {}
  for (const name of session.inputNames) {
    if (!name.startsWith('state_')) continue
    const idx = parseInt(name.replace('state_', ''))
    const { shape, dtype } = FLOW_STATE_SHAPES[idx % 3]
    const isDynamic = shape.some(d => d === 0)
    const s = shape.map(d => d === 0 ? 0 : d)
    if (isDynamic) {
      state[name] = dtype === 'int64'
        ? new ort.Tensor('int64', new BigInt64Array(0), s)
        : new ort.Tensor('float32', new Float32Array(0), s)
    } else {
      const size = s.reduce((a, b) => a * b, 1)
      state[name] = dtype === 'int64'
        ? new ort.Tensor('int64', new BigInt64Array(size), s)
        : new ort.Tensor('float32', new Float32Array(size), s)
    }
  }
  return state
}

function readWavMono24k(wavPath) {
  const buf = readFileSync(wavPath)
  const view = new DataView(buf.buffer, buf.byteOffset, buf.byteLength)
  const sampleRate = view.getUint32(24, true)
  const bitsPerSample = view.getUint16(34, true)
  const numChannels = view.getUint16(22, true)
  const dataSize = view.getUint32(40, true)
  const numSamples = Math.floor(dataSize / (bitsPerSample / 8) / numChannels)
  const samples = new Float32Array(numSamples)
  for (let i = 0; i < numSamples; i++) {
    let sum = 0
    for (let ch = 0; ch < numChannels; ch++) {
      const off = 44 + (i * numChannels + ch) * (bitsPerSample / 8)
      const v = bitsPerSample === 16 ? view.getInt16(off, true) : view.getInt32(off, true)
      sum += bitsPerSample === 16 ? (v < 0 ? v / 0x8000 : v / 0x7FFF) : v / 0x80000000
    }
    samples[i] = sum / numChannels
  }
  const TARGET_RATE = 24000
  const TARGET_LEN = TARGET_RATE * 10
  if (sampleRate === TARGET_RATE) return samples.slice(0, TARGET_LEN)
  const ratio = sampleRate / TARGET_RATE
  const outLen = Math.min(Math.floor(samples.length / ratio), TARGET_LEN)
  const out = new Float32Array(outLen)
  for (let i = 0; i < outLen; i++) {
    const idx = i * ratio
    const lo = Math.floor(idx)
    const hi = Math.min(lo + 1, samples.length - 1)
    out[i] = samples[lo] * (1 - idx + lo) + samples[hi] * (idx - lo)
  }
  return out
}

function saveSafetensors(tensors, outPath) {
  const meta = {}
  let offset = 0
  for (const [name, { data, shape }] of Object.entries(tensors)) {
    meta[name] = { dtype: 'F32', shape, data_offsets: [offset, offset + data.byteLength] }
    offset += data.byteLength
  }
  const headerJson = JSON.stringify(meta)
  const headerBytes = Buffer.from(headerJson, 'utf8')
  const paddedLen = Math.ceil(headerBytes.length / 8) * 8
  const paddedHeader = Buffer.alloc(paddedLen, 0x20)
  headerBytes.copy(paddedHeader)
  const lenBuf = Buffer.alloc(8)
  lenBuf.writeBigUInt64LE(BigInt(paddedLen), 0)
  const parts = [lenBuf, paddedHeader]
  for (const { data } of Object.values(tensors)) {
    parts.push(Buffer.from(data.buffer, data.byteOffset, data.byteLength))
  }
  writeFileSync(outPath, Buffer.concat(parts))
}

const sessionOpts = { executionProviders: ['cpu'], graphOptimizationLevel: 'all' }

console.log('Loading models...')
const [encoder, flowLm] = await Promise.all([
  ort.InferenceSession.create(join(MODEL_DIR, 'mimi_encoder.onnx'), sessionOpts),
  ort.InferenceSession.create(join(MODEL_DIR, 'flow_lm_main_int8.onnx'), sessionOpts),
])
console.log('Models loaded')

const NUM_LAYERS = 6
const wavFiles = readdirSync(VOICES_DIR).filter(f => f.endsWith('.wav'))
console.log('Voices:', wavFiles)

for (const wavFile of wavFiles) {
  const name = wavFile.replace('.wav', '')
  console.log(`\nEncoding ${name}...`)
  const pcm = readWavMono24k(join(VOICES_DIR, wavFile))
  console.log(`  ${pcm.length} samples`)

  const encResult = await encoder.run({ audio: new ort.Tensor('float32', pcm, [1, 1, pcm.length]) })
  const voiceEmb = encResult[encoder.outputNames[0]]
  console.log(`  Embedding: ${voiceEmb.dims}`)

  const state = initFlowState(flowLm)
  const result = await flowLm.run({
    sequence: new ort.Tensor('float32', new Float32Array(0), [1, 0, 32]),
    text_embeddings: voiceEmb,
    ...state,
  })

  const tensors = {}
  for (let layer = 0; layer < NUM_LAYERS; layer++) {
    const cacheOut = result[`out_state_${layer * 3}`]
    const posOut = result[`out_state_${layer * 3 + 2}`]
    const seqLen = Number(posOut.data[0])
    const full = new Float32Array(cacheOut.data)
    const sliced = new Float32Array(2 * seqLen * 16 * 64)
    for (let kv = 0; kv < 2; kv++) {
      for (let t = 0; t < seqLen; t++) {
        for (let h = 0; h < 16; h++) {
          for (let d = 0; d < 64; d++) {
            const src = kv * (1000 * 16 * 64) + t * (16 * 64) + h * 64 + d
            const dst = kv * (seqLen * 16 * 64) + t * (16 * 64) + h * 64 + d
            sliced[dst] = full[src]
          }
        }
      }
    }
    tensors[`transformer.layers.${layer}.self_attn/cache`] = { data: sliced, shape: [2, 1, seqLen, 16, 64] }
    tensors[`transformer.layers.${layer}.self_attn/current_end`] = { data: new Float32Array(seqLen), shape: [seqLen] }
    console.log(`  Layer ${layer}: seqLen=${seqLen}`)
  }

  const outPath = join(OUT_DIR, `${name}.safetensors`)
  saveSafetensors(tensors, outPath)
  console.log(`  Saved: ${outPath} (${statSync(outPath).size} bytes)`)
}
console.log('\nDone!')
