import express from 'express'
import fs from 'fs'
import path from 'path'
import { fileURLToPath } from 'url'
import { createRequire } from 'module'
import { Audio2FaceCore } from './audio2afan_core.mjs'
import ort from 'onnxruntime-node'

const require = createRequire(import.meta.url)
const ttsOnnx = require('webtalk/server-tts-onnx')

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

const CLEETUS_WAV = path.join(__dirname, 'cleetus.wav')
const TTS_MODELS_DIR = path.join(__dirname, 'models', 'tts')

app.get('/', (req, res) => res.sendFile(path.join(__dirname, 'index.html')))
app.get('/client.js', (req, res) => res.sendFile(path.join(__dirname, 'client.js')))
app.get('/Cleetus.vrm', (req, res) => res.sendFile(path.join(__dirname, 'models/Cleetus.vrm')))
app.use('/node_modules', express.static(path.join(__dirname, 'node_modules')))

const SAMPLE_RATE = 24000
const A2F_SAMPLE_RATE = 16000

let a2f = null
let voiceEmbedding = null

async function loadA2F() {
  if (a2f) return a2f
  const audio2afanDir = path.join(__dirname, 'models', 'audio2afan')
  a2f = new Audio2FaceCore({ ort })
  await a2f.loadConfigFile(path.join(audio2afanDir, 'config.json'))
  await a2f.loadModel(path.join(audio2afanDir, 'model.onnx'))
  await a2f.loadSolveData(audio2afanDir)
  console.log('[a2f] Loaded')
  return a2f
}

async function loadVoiceEmbedding() {
  if (voiceEmbedding) return voiceEmbedding
  
  if (!fs.existsSync(CLEETUS_WAV)) {
    console.warn('[voice] No voice file found at', CLEETUS_WAV)
    return null
  }

  console.log('[voice] Encoding voice from', CLEETUS_WAV)
  
  // Load and decode WAV
  const wavBuffer = fs.readFileSync(CLEETUS_WAV)
  const view = new DataView(wavBuffer.buffer, wavBuffer.byteOffset, wavBuffer.byteLength)
  
  // Parse WAV header
  const sampleRate = view.getUint32(24, true)
  const dataSize = view.getUint32(40, true)
  const numSamples = dataSize / 2
  
  const audioData = new Float32Array(numSamples)
  for (let i = 0; i < numSamples; i++) {
    const val = view.getInt16(44 + i * 2, true)
    audioData[i] = val < 0 ? val / 0x8000 : val / 0x7FFF
  }
  
  // Resample to 24kHz if needed
  let resampledData = audioData
  if (sampleRate !== SAMPLE_RATE) {
    resampledData = resampleAudio(audioData, sampleRate, SAMPLE_RATE)
  }
  
  // Load TTS models and encode voice
  await ttsOnnx.loadModels(TTS_MODELS_DIR)
  voiceEmbedding = await ttsOnnx.encodeVoiceAudio(resampledData)
  
  console.log('[voice] Voice encoded, shape:', voiceEmbedding.shape)
  return voiceEmbedding
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

app.post('/api/generate', async (req, res) => {
  try {
    const { text } = req.body
    if (!text) return res.status(400).json({ error: 'text required' })
    
    await loadA2F()
    const voiceEmb = await loadVoiceEmbedding()
    
    if (!voiceEmb) {
      return res.status(500).json({ error: 'Voice embedding not available' })
    }
    
    const startTime = performance.now()
    
    // Generate TTS audio using ONNX
    const audioFloat = await ttsOnnx.synthesize(text, voiceEmb, TTS_MODELS_DIR)
    
    // Encode to WAV
    const audioWav = encodeWAV(audioFloat, SAMPLE_RATE)
    
    // Generate animation
    const audio16k = resampleAudio(audioFloat, SAMPLE_RATE, A2F_SAMPLE_RATE)
    
    const fps = 30
    const stride = Math.round(A2F_SAMPLE_RATE / fps)
    const bufferLen = a2f.bufferLen
    const bufferOfs = a2f.bufferOfs
    const predictionDelay = a2f.faceParams?.prediction_delay || 0
    
    const results = []
    let lastBs = null
    
    for (let offset = 0; offset + bufferLen <= audio16k.length; offset += stride) {
      const chunk = audio16k.slice(offset, offset + bufferLen)
      const result = await a2f.runInference(chunk)
      if (result.blendshapes && lastBs) {
        result.blendshapes = a2f.smoothBlendshapes(lastBs, result.blendshapes)
      }
      lastBs = result.blendshapes
      const centerTime = (offset + bufferOfs) / A2F_SAMPLE_RATE + predictionDelay
      results.push({
        time: centerTime,
        blendshapes: result.blendshapes
      })
    }
    
    const audioDuration = audioFloat.length / SAMPLE_RATE
    const targetNumFrames = Math.ceil(audioDuration * fps)
    const frames = []
    
    for (let frameIdx = 0; frameIdx < targetNumFrames; frameIdx++) {
      const frameTime = frameIdx / fps
      const frame = {}
      
      if (results.length === 0 || frameTime < results[0].time) {
        for (let k = 0; k < 52; k++) {
          frame[ARKIT_NAMES[k]] = 0
        }
      } else {
        let resultIdx = results.findIndex((r, i) => {
          const next = results[i + 1]
          if (!next) return true
          return r.time <= frameTime && next.time > frameTime
        })
        
        if (resultIdx === -1) resultIdx = results.length - 1
        
        const curr = results[resultIdx]
        const next = results[Math.min(resultIdx + 1, results.length - 1)]
        const t = next !== curr && next.time !== curr.time 
          ? (frameTime - curr.time) / (next.time - curr.time) 
          : 0
        
        for (let k = 0; k < 52; k++) {
          const currVal = curr.blendshapes[k].value
          const nextVal = next.blendshapes[k].value
          frame[ARKIT_NAMES[k]] = currVal + (nextVal - currVal) * Math.max(0, Math.min(1, t))
        }
      }
      frames.push(frame)
    }
    
    const animBuffer = buildAfan(frames, fps)
    
    const duration = audioFloat.length / SAMPLE_RATE
    const genTime = ((performance.now() - startTime) / 1000).toFixed(1)
    const rtfx = (duration / parseFloat(genTime)).toFixed(1)
    
    console.log(`[generate] "${text.slice(0, 30)}..." - ${duration.toFixed(1)}s in ${genTime}s (${rtfx}x realtime)`)
    
    res.json({
      audio: audioWav.toString('base64'),
      animation: animBuffer.toString('base64'),
      duration,
    })
  } catch (err) {
    console.error('Generate error:', err)
    res.status(500).json({ error: err.message })
  }
})

async function start() {
  await loadA2F()
  await loadVoiceEmbedding()
  
  app.listen(port, '0.0.0.0', () => {
    console.log(`diagen server running on http://localhost:${port}`)
  })
}

start().catch(err => {
  console.error('Startup failed:', err)
  process.exit(1)
})
