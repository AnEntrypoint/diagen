import { processUserAudio } from './discord-voice-processor.js'
import { pushAudioFrame } from 'dispipe/voice'

const SILENCE_THRESHOLD = 0.01
const SILENCE_DURATION_MS = 1500
const MIN_UTTERANCE_MS = 500
const MAX_UTTERANCE_MS = 30000
const SAMPLE_RATE = 48000

const userBuffers = new Map()
let _processingQueue = null
let _lastError = null

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

function rms(mono) {
  let sum = 0
  for (let i = 0; i < mono.length; i++) sum += mono[i] * mono[i]
  return Math.sqrt(sum / mono.length)
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
    const f32Out = await processUserAudio(pcmBuffer, SAMPLE_RATE, userId)
    pushAudioFrame(f32Out)
    console.log(`[voice] userId=${userId} response sent: ${f32Out.length} samples`)
  } catch (err) {
    console.error(`[voice] userId=${userId} pipeline error: ${err.message}`)
    _lastError.value = { message: err.message, timestamp: Date.now(), userId }
  } finally {
    const idx = _processingQueue.indexOf(entry)
    if (idx !== -1) _processingQueue.splice(idx, 1)
  }
}

export function onPcmChunk(userId, f32) {
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
