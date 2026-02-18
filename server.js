import express from 'express'
import fs from 'fs'
import path from 'path'
import { fileURLToPath } from 'url'
import ort from 'onnxruntime-node'
import { Audio2FaceSDK } from 'audio2afan'

const __dirname = path.dirname(fileURLToPath(import.meta.url))
const app = express()
const port = process.env.PORT || 8080

app.use(express.json({ limit: '50mb' }))

app.use((req, res, next) => {
  res.setHeader('Access-Control-Allow-Origin', '*')
  res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type')
  if (req.method === 'OPTIONS') return res.sendStatus(200)
  next()
})

const TTS_DIR = path.join(__dirname, 'models', 'tts')
const CLEETUS_WAV = path.join(__dirname, 'cleetus.wav')
const CLEETUS_EMB = path.join(__dirname, 'cleetus.emb')

app.get('/', (req, res) => res.sendFile(path.join(__dirname, 'index.html')))
app.get('/client.js', (req, res) => res.sendFile(path.join(__dirname, 'client.js')))
app.get('/Cleetus.vrm', (req, res) => res.sendFile(path.join(__dirname, 'models/Cleetus.vrm')))
app.use('/node_modules', express.static(path.join(__dirname, 'node_modules')))

const SAMPLE_RATE = 24000
const A2F_SAMPLE_RATE = 16000
const SAMPLES_PER_FRAME = 1920
const LSD = 4
const TEMP = 0.7

const sessionOpts = {
  executionProviders: ['cpu'],
  graphOptimizationLevel: 'all',
  enableCpuMemArena: true,
  enableMemPattern: true,
  intraOpNumThreads: 4,
  interOpNumThreads: 4
}

let sessions = null
let stTensors = null
let a2f = null
let voiceEmbedding = null

const FLOW_LM_STATE_SHAPES = {}
for (let i = 0; i < 18; i++) {
  FLOW_LM_STATE_SHAPES[`state_${i}`] = i % 3 === 0 
    ? { shape: [2, 1, 1000, 16, 64], dtype: 'float32' }
    : i % 3 === 1 
      ? { shape: [0], dtype: 'float32' }
      : { shape: [1], dtype: 'int64' }
}

const MIMI_DECODER_STATE_SHAPES = {
  state_0: { shape: [1], dtype: 'bool' },
  state_1: { shape: [1, 512, 6], dtype: 'float32' },
  state_2: { shape: [1], dtype: 'bool' },
  state_3: { shape: [1, 64, 2], dtype: 'float32' },
  state_4: { shape: [1, 256, 6], dtype: 'float32' },
  state_5: { shape: [1], dtype: 'bool' },
  state_6: { shape: [1, 256, 2], dtype: 'float32' },
  state_7: { shape: [1], dtype: 'bool' },
  state_8: { shape: [1, 128, 0], dtype: 'float32' },
  state_9: { shape: [1, 128, 5], dtype: 'float32' },
  state_10: { shape: [1], dtype: 'bool' },
  state_11: { shape: [1, 128, 2], dtype: 'float32' },
  state_12: { shape: [1], dtype: 'bool' },
  state_13: { shape: [1, 64, 0], dtype: 'float32' },
  state_14: { shape: [1, 64, 4], dtype: 'float32' },
  state_15: { shape: [1], dtype: 'bool' },
  state_16: { shape: [1, 64, 2], dtype: 'float32' },
  state_17: { shape: [1], dtype: 'bool' },
  state_18: { shape: [1, 32, 0], dtype: 'float32' },
  state_19: { shape: [2, 1, 8, 1000, 64], dtype: 'float32' },
  state_20: { shape: [1], dtype: 'int64' },
  state_21: { shape: [1], dtype: 'int64' },
  state_22: { shape: [2, 1, 8, 1000, 64], dtype: 'float32' },
  state_23: { shape: [1], dtype: 'int64' },
  state_24: { shape: [1], dtype: 'int64' },
  state_25: { shape: [1], dtype: 'bool' },
  state_26: { shape: [1, 512, 16], dtype: 'float32' },
  state_27: { shape: [1], dtype: 'bool' },
  state_28: { shape: [1, 1, 6], dtype: 'float32' },
  state_29: { shape: [1], dtype: 'bool' },
  state_30: { shape: [1, 64, 2], dtype: 'float32' },
  state_31: { shape: [1], dtype: 'bool' },
  state_32: { shape: [1, 32, 0], dtype: 'float32' },
  state_33: { shape: [1], dtype: 'bool' },
  state_34: { shape: [1, 512, 2], dtype: 'float32' },
  state_35: { shape: [1], dtype: 'bool' },
  state_36: { shape: [1, 64, 4], dtype: 'float32' },
  state_37: { shape: [1], dtype: 'bool' },
  state_38: { shape: [1, 128, 2], dtype: 'float32' },
  state_39: { shape: [1], dtype: 'bool' },
  state_40: { shape: [1, 64, 0], dtype: 'float32' },
  state_41: { shape: [1], dtype: 'bool' },
  state_42: { shape: [1, 128, 5], dtype: 'float32' },
  state_43: { shape: [1], dtype: 'bool' },
  state_44: { shape: [1, 256, 2], dtype: 'float32' },
  state_45: { shape: [1], dtype: 'bool' },
  state_46: { shape: [1, 128, 0], dtype: 'float32' },
  state_47: { shape: [1], dtype: 'bool' },
  state_48: { shape: [1, 256, 6], dtype: 'float32' },
  state_49: { shape: [2, 1, 8, 1000, 64], dtype: 'float32' },
  state_50: { shape: [1], dtype: 'int64' },
  state_51: { shape: [1], dtype: 'int64' },
  state_52: { shape: [2, 1, 8, 1000, 64], dtype: 'float32' },
  state_53: { shape: [1], dtype: 'int64' },
  state_54: { shape: [1], dtype: 'int64' },
  state_55: { shape: [1, 512, 16], dtype: 'float32' },
}

function initState(session, stateShapes) {
  const state = {}
  for (const inputName of session.inputNames) {
    if (inputName.startsWith('state_')) {
      const stateInfo = stateShapes[inputName]
      if (!stateInfo) continue
      let { shape, dtype } = stateInfo
      const isDynamic = shape.some(d => d === 0)
      if (isDynamic) {
        const emptyShape = shape.map(d => d === 0 ? 0 : d)
        state[inputName] = dtype === 'int64' 
          ? new ort.Tensor('int64', new BigInt64Array(0), emptyShape)
          : dtype === 'bool'
            ? new ort.Tensor('bool', new Uint8Array(0), emptyShape)
            : new ort.Tensor('float32', new Float32Array(0), emptyShape)
      } else {
        const size = shape.reduce((a, b) => a * b, 1)
        state[inputName] = dtype === 'int64' 
          ? new ort.Tensor('int64', new BigInt64Array(size), shape)
          : dtype === 'bool'
            ? new ort.Tensor('bool', new Uint8Array(size), shape)
            : new ort.Tensor('float32', new Float32Array(size), shape)
      }
    }
  }
  return state
}

function initStTensors() {
  stTensors = {}
  const dt = 1.0 / LSD
  stTensors[LSD] = []
  for (let j = 0; j < LSD; j++) {
    const s = j / LSD
    const t = s + dt
    stTensors[LSD].push({
      s: new ort.Tensor('float32', new Float32Array([s]), [1, 1]),
      t: new ort.Tensor('float32', new Float32Array([t]), [1, 1])
    })
  }
}

async function loadModels() {
  if (sessions) return sessions
  
  const files = {
    mimiEncoder: path.join(TTS_DIR, 'mimi_encoder.onnx'),
    textConditioner: path.join(TTS_DIR, 'text_conditioner.onnx'),
    flowLmMain: path.join(TTS_DIR, 'flow_lm_main_int8.onnx'),
    flowLmFlow: path.join(TTS_DIR, 'flow_lm_flow_int8.onnx'),
    mimiDecoder: path.join(TTS_DIR, 'mimi_decoder_int8.onnx'),
  }
  
  console.log('[models] Loading...')
  sessions = {
    mimiEncoder: await ort.InferenceSession.create(files.mimiEncoder, sessionOpts),
    textConditioner: await ort.InferenceSession.create(files.textConditioner, sessionOpts),
    flowLmMain: await ort.InferenceSession.create(files.flowLmMain, sessionOpts),
    flowLmFlow: await ort.InferenceSession.create(files.flowLmFlow, sessionOpts),
    mimiDecoder: await ort.InferenceSession.create(files.mimiDecoder, sessionOpts),
  }
  
  const audio2afanDir = path.join(__dirname, 'models', 'audio2afan')
  a2f = new Audio2FaceSDK()
  await a2f.loadConfigFile(path.join(audio2afanDir, 'config.json'))
  await a2f.loadModel(path.join(audio2afanDir, 'model.onnx'))
  
  initStTensors()
  await loadVoiceEmbedding()
  console.log('[models] Loaded')
  return sessions
}

function encodeWAV(float32Data, sampleRate) {
  const wavBuf = new ArrayBuffer(44 + float32Data.length * 2)
  const view = new DataView(wavBuf)
  const writeStr = (o, s) => { for (let i = 0; i < s.length; i++) view.setUint8(o + i, s.charCodeAt(i)) }
  writeStr(0, 'RIFF')
  view.setUint32(4, 36 + float32Data.length * 2, true)
  writeStr(8, 'WAVE')
  writeStr(12, 'fmt ')
  view.setUint32(16, 16, true)
  view.setUint16(20, 1, true)
  view.setUint16(22, 1, true)
  view.setUint32(24, sampleRate, true)
  view.setUint32(28, sampleRate * 2, true)
  view.setUint16(32, 2, true)
  view.setUint16(34, 16, true)
  writeStr(36, 'data')
  view.setUint32(40, float32Data.length * 2, true)
  for (let i = 0; i < float32Data.length; i++) {
    const s = Math.max(-1, Math.min(1, float32Data[i]))
    view.setInt16(44 + i * 2, s < 0 ? s * 0x8000 : s * 0x7FFF, true)
  }
  return Buffer.from(wavBuf)
}

function decodeWAV(wavBuffer) {
  const view = new DataView(wavBuffer.buffer || wavBuffer)
  const readStr = (o, len) => {
    let s = ''
    for (let i = 0; i < len; i++) s += String.fromCharCode(view.getUint8(o + i))
    return s
  }
  if (readStr(0, 4) !== 'RIFF' || readStr(8, 4) !== 'WAVE') {
    throw new Error('Invalid WAV file')
  }
  const sampleRate = view.getUint32(24, true)
  const bitsPerSample = view.getUint16(34, true)
  const dataOffset = 44
  const dataSize = view.getUint32(40, true)
  const numSamples = dataSize / (bitsPerSample / 8)
  const float32Data = new Float32Array(numSamples)
  for (let i = 0; i < numSamples; i++) {
    const val = view.getInt16(dataOffset + i * 2, true)
    float32Data[i] = val < 0 ? val / 0x8000 : val / 0x7FFF
  }
  return { data: float32Data, sampleRate }
}

async function encodeVoiceFromWAV(wavPath) {
  const wavData = fs.readFileSync(wavPath)
  const { data, sampleRate } = decodeWAV(wavData)
  let audioData = data
  if (sampleRate !== SAMPLE_RATE) {
    audioData = resampleAudio(data, sampleRate, SAMPLE_RATE)
  }
  const input = new ort.Tensor('float32', audioData, [1, 1, audioData.length])
  const result = await sessions.mimiEncoder.run({ audio: input })
  const emb = result[sessions.mimiEncoder.outputNames[0]]
  return { data: new Float32Array(emb.data), shape: emb.dims }
}

async function loadVoiceEmbedding() {
  if (voiceEmbedding) return voiceEmbedding
  if (fs.existsSync(CLEETUS_EMB)) {
    console.log('[voice] Loading cached embedding from', CLEETUS_EMB)
    const embData = fs.readFileSync(CLEETUS_EMB)
    const view = new DataView(embData.buffer, embData.byteOffset, embData.byteLength)
    const numFrames = view.getUint32(0, true)
    const embDim = view.getUint32(4, true)
    const data = new Float32Array(embData.buffer, embData.byteOffset + 8, numFrames * embDim)
    voiceEmbedding = { data: new Float32Array(data), shape: [1, numFrames, embDim] }
  } else if (fs.existsSync(CLEETUS_WAV)) {
    console.log('[voice] Encoding voice from', CLEETUS_WAV)
    voiceEmbedding = await encodeVoiceFromWAV(CLEETUS_WAV)
    const numFrames = voiceEmbedding.shape[1]
    const embDim = voiceEmbedding.shape[2]
    const embBuffer = Buffer.alloc(8 + voiceEmbedding.data.length * 4)
    embBuffer.writeUInt32LE(numFrames, 0)
    embBuffer.writeUInt32LE(embDim, 4)
    const dataView = new DataView(embBuffer.buffer, embBuffer.byteOffset + 8, voiceEmbedding.data.length * 4)
    for (let i = 0; i < voiceEmbedding.data.length; i++) {
      dataView.setFloat32(i * 4, voiceEmbedding.data[i], true)
    }
    fs.writeFileSync(CLEETUS_EMB, embBuffer)
    console.log('[voice] Cached embedding to', CLEETUS_EMB)
  } else {
    console.warn('[voice] No voice file found, using default')
    voiceEmbedding = null
  }
  return voiceEmbedding
}

function resampleAudio(float32Data, fromRate, toRate) {
  const ratio = fromRate / toRate
  const newLen = Math.round(float32Data.length / ratio)
  const result = new Float32Array(newLen)
  for (let i = 0; i < newLen; i++) {
    const idx = i * ratio
    const lo = Math.floor(idx)
    const hi = Math.min(lo + 1, float32Data.length - 1)
    const frac = idx - lo
    result[i] = float32Data[lo] * (1 - frac) + float32Data[hi] * frac
  }
  return result
}

const ARKIT_NAMES = [
  'browInnerUp', 'browDownLeft', 'browDownRight', 'browOuterUpLeft', 'browOuterUpRight',
  'eyeLookUpLeft', 'eyeLookUpRight', 'eyeLookDownLeft', 'eyeLookDownRight',
  'eyeLookInLeft', 'eyeLookInRight', 'eyeLookOutLeft', 'eyeLookOutRight',
  'eyeBlinkLeft', 'eyeBlinkRight', 'eyeSquintLeft', 'eyeSquintRight',
  'eyeWideLeft', 'eyeWideRight', 'cheekPuff', 'cheekSquintLeft', 'cheekSquintRight',
  'noseSneerLeft', 'noseSneerRight', 'jawOpen', 'jawForward', 'jawLeft', 'jawRight',
  'mouthFunnel', 'mouthPucker', 'mouthLeft', 'mouthRight',
  'mouthRollUpper', 'mouthRollLower', 'mouthShrugUpper', 'mouthShrugLower',
  'mouthOpen', 'mouthClose', 'mouthSmileLeft', 'mouthSmileRight',
  'mouthFrownLeft', 'mouthFrownRight', 'mouthDimpleLeft', 'mouthDimpleRight',
  'mouthUpperUpLeft', 'mouthUpperUpRight', 'mouthLowerDownLeft', 'mouthLowerDownRight',
  'mouthPressLeft', 'mouthPressRight', 'mouthStretchLeft', 'mouthStretchRight'
]

function buildAfan(frames, fps = 30) {
  const numBlendshapes = ARKIT_NAMES.length
  const numFrames = frames.length
  const totalSize = 12 + (numFrames * numBlendshapes)
  const buf = Buffer.alloc(totalSize)
  let offset = 0
  buf.writeUInt32LE(0x4146414E, offset); offset += 4
  buf.writeUInt8(2, offset); offset += 1
  buf.writeUInt8(fps, offset); offset += 1
  buf.writeUInt8(numBlendshapes, offset); offset += 1
  buf.writeUInt8(0, offset); offset += 1
  buf.writeUInt32LE(numFrames, offset); offset += 4
  for (const frame of frames) {
    for (let i = 0; i < numBlendshapes; i++) {
      buf[offset++] = Math.round(Math.max(0, Math.min(1, frame[ARKIT_NAMES[i]] || 0)) * 255)
    }
  }
  return buf
}

const RAND_BUF = new Float32Array(64)
function fillRand() {
  for (let i = 0; i < 64; i += 2) {
    let u = 0, v = 0
    while (u === 0) u = Math.random()
    while (v === 0) v = Math.random()
    RAND_BUF[i] = Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v)
    RAND_BUF[i + 1] = Math.sqrt(-2.0 * Math.log(u)) * Math.sin(2.0 * Math.PI * v)
  }
}

class Profiler {
  constructor() {
    this.timings = {}
    this.starts = {}
  }
  start(name) {
    this.starts[name] = Date.now()
  }
  end(name) {
    if (!this.starts[name]) return
    const elapsed = Date.now() - this.starts[name]
    this.timings[name] = (this.timings[name] || 0) + elapsed
    delete this.starts[name]
    return elapsed
  }
  report() {
    const total = Object.values(this.timings).reduce((a, b) => a + b, 0)
    const lines = Object.entries(this.timings)
      .sort((a, b) => b[1] - a[1])
      .map(([k, v]) => `  ${k}: ${v}ms (${((v/total)*100).toFixed(1)}%)`)
    return lines.join('\n')
  }
}

async function generateTTS(text, prof) {
  const s = sessions
  
  const MAX_FRAMES = Math.min(500, Math.max(50, Math.ceil(text.length * 10)))
  
  prof.start('tokenize')
  const tokens = text.toLowerCase().replace(/[^a-z0-9\s.,!?']/g, '').split(/\s+/).filter(Boolean)
  const tokenIds = tokens.flatMap(t => Array.from(t).map(c => c.charCodeAt(0)))
  if (tokenIds.length === 0) tokenIds.push(32)
  prof.end('tokenize')
  
  prof.start('text_condition')
  const textTensor = new ort.Tensor('int64', BigInt64Array.from(tokenIds.map(x => BigInt(x))), [1, tokenIds.length])
  const textResult = await s.textConditioner.run({ token_ids: textTensor })
  let textEmb = textResult[s.textConditioner.outputNames[0]]
  if (textEmb.dims.length === 2) {
    textEmb = new ort.Tensor('float32', textEmb.data, [1, textEmb.dims[0], textEmb.dims[1]])
  }
  prof.end('text_condition')
  
  prof.start('voice_condition')
  let flowLmState = initState(s.flowLmMain, FLOW_LM_STATE_SHAPES)
  const emptySeq = new ort.Tensor('float32', new Float32Array(32).fill(NaN), [1, 1, 32])
  
  if (voiceEmbedding) {
    const voiceTensor = new ort.Tensor('float32', voiceEmbedding.data, voiceEmbedding.shape)
    const voiceCondResult = await s.flowLmMain.run({ sequence: emptySeq, text_embeddings: voiceTensor, ...flowLmState })
    for (let i = 2; i < s.flowLmMain.outputNames.length; i++) {
      const outputName = s.flowLmMain.outputNames[i]
      if (outputName.startsWith('out_state_')) {
        flowLmState[`state_${parseInt(outputName.replace('out_state_', ''))}`] = voiceCondResult[outputName]
      }
    }
  }
  prof.end('voice_condition')
  
  prof.start('flow_init')
  const textCondResult = await s.flowLmMain.run({ sequence: emptySeq, text_embeddings: textEmb, ...flowLmState })
  const conditioning = textCondResult['conditioning']
  
  for (let i = 2; i < s.flowLmMain.outputNames.length; i++) {
    const outputName = s.flowLmMain.outputNames[i]
    if (outputName.startsWith('out_state_')) {
      flowLmState[`state_${parseInt(outputName.replace('out_state_', ''))}`] = textCondResult[outputName]
    }
  }
  prof.end('flow_init')
  
  prof.start('flow_ar')
  const latents = []
  const STD = Math.sqrt(TEMP)
  const dt = 1.0 / LSD
  const stArr = stTensors[LSD]
  const xData = new Float32Array(32)
  
  for (let step = 0; step < MAX_FRAMES; step++) {
    fillRand()
    for (let i = 0; i < 32; i++) xData[i] = RAND_BUF[i] * STD
    
    for (let j = 0; j < LSD; j++) {
      const x_tensor = new ort.Tensor('float32', xData, [1, 32])
      const flowResult = await s.flowLmFlow.run({ c: conditioning, s: stArr[j].s, t: stArr[j].t, x: x_tensor })
      const v = flowResult['flow_dir'].data
      for (let k = 0; k < 32; k++) xData[k] += v[k] * dt
    }
    
    latents.push(new Float32Array(xData))
  }
  prof.end('flow_ar')
  
  prof.start('decoder')
  const BATCH_SIZE = 50
  const audioChunks = []
  let mimiState = initState(s.mimiDecoder, MIMI_DECODER_STATE_SHAPES)
  
  for (let i = 0; i < latents.length; i += BATCH_SIZE) {
    const batchLen = Math.min(BATCH_SIZE, latents.length - i)
    const latentData = new Float32Array(batchLen * 32)
    for (let j = 0; j < batchLen; j++) {
      latentData.set(latents[i + j], j * 32)
    }
    
    const latentTensor = new ort.Tensor('float32', latentData, [1, batchLen, 32])
    const decResult = await s.mimiDecoder.run({ latent: latentTensor, ...mimiState })
    audioChunks.push(new Float32Array(decResult[s.mimiDecoder.outputNames[0]].data))
    
    for (let k = 1; k < s.mimiDecoder.outputNames.length; k++) {
      const outputName = s.mimiDecoder.outputNames[k]
      if (outputName.startsWith('state_') || outputName.startsWith('out_state_')) {
        mimiState[outputName.replace('out_', '')] = decResult[outputName]
      }
    }
  }
  prof.end('decoder')
  
  return audioChunks
}

app.post('/api/generate', async (req, res) => {
  try {
    const { text } = req.body
    if (!text) return res.status(400).json({ error: 'text required' })
    
    await loadModels()
    
    const prof = new Profiler()
    prof.start('total')
    
    const audioChunks = await generateTTS(text, prof)
    
    prof.start('merge_audio')
    const totalLen = audioChunks.reduce((s, c) => s + c.length, 0)
    const audioFloat = new Float32Array(totalLen)
    let off = 0
    for (const chunk of audioChunks) {
      audioFloat.set(chunk, off)
      off += chunk.length
    }
    prof.end('merge_audio')
    
    prof.start('resample')
    const audio16k = resampleAudio(audioFloat, SAMPLE_RATE, A2F_SAMPLE_RATE)
    prof.end('resample')
    
    prof.start('a2f')
    const results = []
    const bufferLen = a2f.bufferLen
    const bufferOfs = a2f.bufferOfs
    let lastBs = null
    
    const a2fPromises = []
    for (let i = 0; i < audio16k.length - bufferLen; i += bufferOfs) {
      const chunk = audio16k.slice(i, i + bufferLen)
      a2fPromises.push(a2f.runInference(chunk))
    }
    
    const a2fResults = await Promise.all(a2fPromises)
    for (let i = 0; i < a2fResults.length; i++) {
      const result = a2fResults[i]
      if (result.blendshapes && lastBs) {
        result.blendshapes = a2f.smoothBlendshapes(lastBs, result.blendshapes)
      }
      lastBs = result.blendshapes
      results.push(result)
    }
    prof.end('a2f')
    
    prof.start('build_output')
    const fps = 30
    const frames = results.map(r => {
      const frame = {}
      for (const bs of r.blendshapes) frame[bs.name] = bs.value
      return frame
    })
    
    const animBuffer = buildAfan(frames, fps)
    const audioWav = encodeWAV(audioFloat, SAMPLE_RATE)
    prof.end('build_output')
    
    prof.end('total')
    
    const duration = audioFloat.length / SAMPLE_RATE
    const totalMs = prof.timings.total
    console.log(`\n[profile] "${text.slice(0, 30)}..."`)
    console.log(`Duration: ${duration.toFixed(1)}s, Total: ${totalMs}ms, RTFx: ${(duration / (totalMs/1000)).toFixed(2)}x`)
    console.log(prof.report())
    
    res.json({
      audio: audioWav.toString('base64'),
      animation: animBuffer.toString('base64'),
      duration,
      profile: prof.timings
    })
  } catch (err) {
    console.error('Generate error:', err)
    res.status(500).json({ error: err.message })
  }
})

loadModels().then(() => {
  app.listen(port, '0.0.0.0', () => {
    console.log(`diagen server running on http://localhost:${port}`)
  })
}).catch(err => {
  console.error('Startup failed:', err)
  process.exit(1)
})
