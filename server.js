import express from 'express'
import fs from 'fs'
import path from 'path'
import { fileURLToPath } from 'url'
import { createRequire } from 'module'
import { Audio2FaceCore } from './audio2afan_core.mjs'
import ort from 'onnxruntime-node'
import { ARKIT_NAMES, encodeWAV, resampleAudio, buildAfan } from './server-utils.mjs'
import os from 'os'

const require = createRequire(import.meta.url)
const ORT_CPUS = os.cpus().length
const origOrtCreate = ort.InferenceSession.create.bind(ort.InferenceSession)
ort.InferenceSession.create = async function(modelPath, options = {}) {
  const providers = options.executionProviders || ['cpu']
  const patched = {
    ...options,
    executionProviders: providers.includes('dml') ? providers : ['dml', ...providers],
    intraOpNumThreads: Math.max(options.intraOpNumThreads || 0, ORT_CPUS),
    interOpNumThreads: Math.max(options.interOpNumThreads || 0, Math.floor(ORT_CPUS / 4)),
    executionMode: 'parallel',
    enableCpuMemArena: true,
    enableMemPattern: true,
    graphOptimizationLevel: options.graphOptimizationLevel || 'all',
  }
  return origOrtCreate(modelPath, patched)
}
const ttsOnnx = require('webtalk/server-tts-onnx')

const __dirname = path.dirname(fileURLToPath(import.meta.url))
const app = express()
const port = process.env.PORT || 8080
app.use(express.json({ limit: '50mb' }))
app.use((req, res, next) => {
  res.setHeader('Cross-Origin-Opener-Policy', 'same-origin')
  res.setHeader('Cross-Origin-Embedder-Policy', 'require-corp')
  res.setHeader('Access-Control-Allow-Origin', '*')
  res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type')
  if (req.method === 'OPTIONS') return res.sendStatus(200)
  next()
})
const CLEETUS_WAV = path.join(__dirname, 'voices', 'cleetus.wav')
const TTS_MODELS_DIR = path.join(__dirname, 'models', 'tts')
app.get('/', (req, res) => res.sendFile(path.join(__dirname, 'index.html')))
app.get('/client.js', (req, res) => res.sendFile(path.join(__dirname, 'client.js')))
app.get('/animation-core.mjs', (req, res) => res.sendFile(path.join(__dirname, 'animation-core.mjs')))
app.get('/idle-animator.mjs', (req, res) => res.sendFile(path.join(__dirname, 'idle-animator.mjs')))
app.get('/facial-player.mjs', (req, res) => res.sendFile(path.join(__dirname, 'facial-player.mjs')))
app.get('/llm-worker.js', (req, res) => res.sendFile(path.join(__dirname, 'llm-worker.js')))
app.get('/Cleetus.vrm', (req, res) => res.sendFile(path.join(__dirname, 'Cleetus.vrm')))
app.use('/node_modules', express.static(path.join(__dirname, 'node_modules')))
const DEMO_DIR = path.join(__dirname, 'gh-pages-src', 'demo')
const VOICES_DIR = path.join(__dirname, 'voices')
app.get('/demo/voices/manifest.json', (req, res) => {
  const files = fs.existsSync(VOICES_DIR) ? fs.readdirSync(VOICES_DIR).filter(f => f.endsWith('.wav')) : []
  res.json(files)
})
app.use('/demo/voices', express.static(VOICES_DIR))
app.use('/demo', express.static(DEMO_DIR))
const SAMPLE_RATE = 24000, A2F_SAMPLE_RATE = 16000
let a2f = null, voiceEmbedding = null

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
  const wavBuffer = fs.readFileSync(CLEETUS_WAV)
  const view = new DataView(wavBuffer.buffer, wavBuffer.byteOffset, wavBuffer.byteLength)
  const sampleRate = view.getUint32(24, true)
  const dataSize = view.getUint32(40, true)
  const numSamples = dataSize / 2
  const audioData = new Float32Array(numSamples)
  for (let i = 0; i < numSamples; i++) {
    const val = view.getInt16(44 + i * 2, true)
    audioData[i] = val < 0 ? val / 0x8000 : val / 0x7FFF
  }
  let resampledData = audioData
  if (sampleRate !== SAMPLE_RATE) {
    resampledData = resampleAudio(audioData, sampleRate, SAMPLE_RATE)
  }
  await ttsOnnx.loadModels(TTS_MODELS_DIR)
  voiceEmbedding = await ttsOnnx.encodeVoiceAudio(resampledData)
  console.log('[voice] Voice encoded, shape:', voiceEmbedding.shape)
  return voiceEmbedding
}
app.post('/api/generate', async (req, res) => {
  try {
    const { text } = req.body
    if (!text) return res.status(400).json({ error: 'text required' })
    await loadA2F()
    const voiceEmb = await loadVoiceEmbedding()
    if (!voiceEmb) return res.status(500).json({ error: 'Voice embedding not available' })
    const startTime = performance.now()

    const audioFloat = await ttsOnnx.synthesize(text, voiceEmb, TTS_MODELS_DIR)
    const [audioWav, audio16k] = await Promise.all([
      Promise.resolve(encodeWAV(audioFloat, SAMPLE_RATE)),
      Promise.resolve(resampleAudio(audioFloat, SAMPLE_RATE, A2F_SAMPLE_RATE))
    ])
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
      results.push({
        time: (offset + bufferOfs) / A2F_SAMPLE_RATE + predictionDelay,
        blendshapes: result.blendshapes
      })
    }
    const audioDuration = audioFloat.length / SAMPLE_RATE
    const targetNumFrames = Math.ceil(audioDuration * fps)
    const frames = new Array(targetNumFrames)
    let cursor = 0
    for (let frameIdx = 0; frameIdx < targetNumFrames; frameIdx++) {
      const frameTime = frameIdx / fps
      const frame = new Float32Array(52)
      if (results.length === 0 || frameTime < results[0].time) { frames[frameIdx] = frame; continue }
      while (cursor < results.length - 1 && results[cursor + 1].time <= frameTime) cursor++

      const curr = results[cursor]
      const next = results[Math.min(cursor + 1, results.length - 1)]
      const dt = next.time - curr.time
      const t = dt > 0 ? Math.max(0, Math.min(1, (frameTime - curr.time) / dt)) : 0

      for (let k = 0; k < 52; k++) {
        frame[k] = curr.blendshapes[k].value + (next.blendshapes[k].value - curr.blendshapes[k].value) * t
      }
      frames[frameIdx] = frame
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
async function ensureModels() {
  const { downloadModels } = await import('./download-models.js')
  await downloadModels()
}
async function start() {
  await ensureModels()
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
