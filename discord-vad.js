import { processUserAudio } from './discord-voice-processor.js'
import { pushAudioFrame } from 'dispipe/voice'

const SILENCE_THRESHOLD = 0.0005
const SILENCE_DURATION_MS = 1500
const MIN_UTTERANCE_MS = 500
const MAX_UTTERANCE_MS = 30000
const SAMPLE_RATE = 48000

const userBuffers = new Map()
let _processingQueue = null
let _lastError = null
let _botSpeakingUntil = 0
const BOT_SPEAK_TAIL_MS = 800

export function init(processingQueue, lastErrorRef) {
  _processingQueue = processingQueue
  _lastError = lastErrorRef
}

export function getBuffers() { return userBuffers }

function getOrCreateBuffer(userId) {
  if (!userBuffers.has(userId)) {
    userBuffers.set(userId, { chunks: [], startTime: 0, lastVoiceTime: 0, processing: false })
  }
  return userBuffers.get(userId)
}

function rms(samples) {
  let sum = 0
  for (let i = 0; i < samples.length; i++) sum += samples[i] * samples[i]
  return Math.sqrt(sum / samples.length)
}

async function handleUtterance(userId, chunks) {
  const totalLen = chunks.reduce((s, c) => s + c.length, 0)
  const merged = new Float32Array(totalLen)
  let offset = 0
  for (const chunk of chunks) { merged.set(chunk, offset); offset += chunk.length }
  const int16 = new Int16Array(merged.length)
  for (let i = 0; i < merged.length; i++) {
    const v = Math.max(-1, Math.min(1, merged[i]))
    int16[i] = v < 0 ? v * 0x8000 : v * 0x7FFF
  }
  const pcmBuffer = Buffer.from(int16.buffer)
  console.log(`[voice] userId=${userId} utterance: ${(totalLen / SAMPLE_RATE).toFixed(1)}s, ${totalLen} samples`)

  const entry = { userId, startTime: Date.now(), samples: totalLen }
  _processingQueue.push(entry)
  try {
    const monoOut = await processUserAudio(pcmBuffer, SAMPLE_RATE, userId)
    if (!monoOut) return
    const stereo = new Float32Array(monoOut.length * 2)
    for (let i = 0; i < monoOut.length; i++) { stereo[i * 2] = monoOut[i]; stereo[i * 2 + 1] = monoOut[i] }
    const durationMs = (monoOut.length / SAMPLE_RATE) * 1000
    _botSpeakingUntil = Date.now() + durationMs + BOT_SPEAK_TAIL_MS
    pushAudioFrame(stereo)
    console.log(`[voice] userId=${userId} response sent: ${monoOut.length} mono samples, bot-speaking ${durationMs.toFixed(0)}ms`)
    for (const b of userBuffers.values()) b.chunks = []
  } catch (err) {
    console.error(`[voice] userId=${userId} pipeline error: ${err.message}`)
    _lastError.value = { message: err.message, timestamp: Date.now(), userId }
  } finally {
    const idx = _processingQueue.indexOf(entry)
    if (idx !== -1) _processingQueue.splice(idx, 1)
  }
}

export function onPcmChunk(userId, stereoF32) {
  if (Date.now() < _botSpeakingUntil) return
  const monoLen = stereoF32.length / 2
  const f32 = new Float32Array(monoLen)
  for (let i = 0; i < monoLen; i++) f32[i] = (stereoF32[i * 2] + stereoF32[i * 2 + 1]) * 0.5

  const buf = getOrCreateBuffer(userId)
  if (buf.processing) return

  const now = Date.now()
  const level = rms(f32)
  const isSpeech = level > SILENCE_THRESHOLD

  if (isSpeech) {
    if (buf.chunks.length === 0) buf.startTime = now
    buf.lastVoiceTime = now
    buf.chunks.push(f32)
  }

  if (buf.chunks.length === 0) return

  const utteranceDuration = now - buf.startTime
  const silenceDuration = now - buf.lastVoiceTime
  const shouldFlush = silenceDuration >= SILENCE_DURATION_MS || utteranceDuration >= MAX_UTTERANCE_MS

  if (!shouldFlush) return
  if (utteranceDuration < MIN_UTTERANCE_MS) { buf.chunks = []; return }

  const chunks = buf.chunks
  buf.chunks = []
  buf.processing = true
  handleUtterance(userId, chunks).finally(() => { buf.processing = false })
}
