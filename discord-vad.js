import { processUserAudio } from './discord-voice-processor.js'
import { pushAudioFrame, flushAudio } from 'dispipe/voice'

const SILENCE_THRESHOLD_BASE = 0.004
const INTERRUPT_THRESHOLD = 0.025
const SILENCE_DURATION_MS = 900
const INTERRUPT_SILENCE_MS = 350
const MIN_UTTERANCE_MS = 400
const MIN_SPEECH_SAMPLES = 48000 * 0.4
const MAX_UTTERANCE_MS = 15000
const SAMPLE_RATE = 48000
const BOT_SPEAK_TAIL_MS = 250
const PREROLL_MS = 500
const PREROLL_SAMPLES = SAMPLE_RATE * PREROLL_MS / 1000
const MIN_PEAK_RMS = 0.015

const userBuffers = new Map()
let _processingQueue = null
let _lastError = null
let _botSpeakingUntil = 0
let _currentAbort = null
let _activeUtteranceCount = 0
let _usernameResolver = (uid) => `user${String(uid).slice(-4)}`

export function setUsernameResolver(fn) { _usernameResolver = fn }

export function init(processingQueue, lastErrorRef) {
  _processingQueue = processingQueue
  _lastError = lastErrorRef
  console.log(`[vad] init threshold=${SILENCE_THRESHOLD_BASE} interrupt=${INTERRUPT_THRESHOLD} silence=${SILENCE_DURATION_MS}ms min=${MIN_UTTERANCE_MS}ms preroll=${PREROLL_MS}ms`)
}

export function getBuffers() { return userBuffers }

function getOrCreateBuffer(userId) {
  if (!userBuffers.has(userId)) {
    userBuffers.set(userId, { chunks: [], preroll: [], prerollSamples: 0, startTime: 0, lastVoiceTime: 0, processing: false, peakRms: 0, capturing: false })
    console.log(`[vad] new buffer for uid=${userId}`)
  }
  return userBuffers.get(userId)
}

function rms(samples) {
  let sum = 0
  for (let i = 0; i < samples.length; i++) sum += samples[i] * samples[i]
  return Math.sqrt(sum / samples.length)
}

async function handleUtterance(userId, chunks, peakRms) {
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
  const durSec = (totalLen / SAMPLE_RATE).toFixed(2)
  _activeUtteranceCount++
  console.log(`[vad] ▶ utterance uid=${userId} dur=${durSec}s peak=${peakRms.toFixed(4)} samples=${totalLen} active=${_activeUtteranceCount}`)

  const abort = new AbortController()
  _currentAbort = abort
  const entry = { userId, startTime: Date.now(), samples: totalLen }
  _processingQueue.push(entry)
  try {
    const username = _usernameResolver(userId)
    const monoOut = await processUserAudio(pcmBuffer, SAMPLE_RATE, userId, abort.signal, username)
    if (!monoOut || abort.signal.aborted) {
      console.log(`[vad] uid=${userId} no output (aborted=${abort.signal.aborted})`)
      return
    }
    const stereo = new Float32Array(monoOut.length * 2)
    for (let i = 0; i < monoOut.length; i++) { stereo[i * 2] = monoOut[i]; stereo[i * 2 + 1] = monoOut[i] }
    const durationMs = (monoOut.length / SAMPLE_RATE) * 1000
    _botSpeakingUntil = Date.now() + durationMs + BOT_SPEAK_TAIL_MS
    pushAudioFrame(stereo)
    console.log(`[vad] 📢 speaking uid=${userId} ${durationMs.toFixed(0)}ms → until ${new Date(_botSpeakingUntil).toISOString().slice(14,23)}`)
  } catch (err) {
    if (err.name === 'AbortError' || err.message?.includes('aborted')) {
      console.log(`[vad] uid=${userId} pipeline aborted cleanly`)
    } else {
      console.error(`[vad] ✗ uid=${userId} pipeline error:`, err.stack || err.message)
      _lastError.value = { message: err.message, timestamp: Date.now(), userId }
    }
  } finally {
    if (_currentAbort === abort) _currentAbort = null
    const idx = _processingQueue.indexOf(entry)
    if (idx !== -1) _processingQueue.splice(idx, 1)
    const buf = userBuffers.get(userId)
    if (buf) buf.processing = false
    _activeUtteranceCount--
    const elapsed = Date.now() - entry.startTime
    console.log(`[vad] ◀ done uid=${userId} totalMs=${elapsed} active=${_activeUtteranceCount}`)
  }
}

export function onPcmChunk(userId, stereoF32) {
  const now = Date.now()
  const botSpeaking = now < _botSpeakingUntil
  const monoLen = stereoF32.length / 2
  const f32 = new Float32Array(monoLen)
  for (let i = 0; i < monoLen; i++) f32[i] = (stereoF32[i * 2] + stereoF32[i * 2 + 1]) * 0.5

  const buf = getOrCreateBuffer(userId)
  if (buf.processing) return

  const level = rms(f32)
  const effectiveThreshold = botSpeaking ? INTERRUPT_THRESHOLD : SILENCE_THRESHOLD_BASE
  const isSpeech = level > effectiveThreshold

  if (!buf.capturing) {
    buf.preroll.push(f32)
    buf.prerollSamples += f32.length
    while (buf.prerollSamples > PREROLL_SAMPLES && buf.preroll.length > 1) {
      buf.prerollSamples -= buf.preroll[0].length
      buf.preroll.shift()
    }
    if (!isSpeech) return
    buf.capturing = true
    buf.startTime = now - (buf.prerollSamples / SAMPLE_RATE) * 1000
    buf.lastVoiceTime = now
    buf.peakRms = level
    buf.chunks = [...buf.preroll]
    buf.preroll = []
    buf.prerollSamples = 0
    console.log(`[vad] 🎤 speech-start uid=${userId} rms=${level.toFixed(4)} botSpeaking=${botSpeaking} preroll=${buf.chunks.reduce((s,c)=>s+c.length,0)}samples`)
  } else {
    buf.chunks.push(f32)
    if (isSpeech) {
      if (level > buf.peakRms) buf.peakRms = level
      buf.lastVoiceTime = now
    }
  }

  if (!buf.capturing) return

  const utteranceDuration = now - buf.startTime
  const silenceDuration = now - buf.lastVoiceTime
  const flushSilence = botSpeaking ? INTERRUPT_SILENCE_MS : SILENCE_DURATION_MS
  const shouldFlush = silenceDuration >= flushSilence || utteranceDuration >= MAX_UTTERANCE_MS

  if (!shouldFlush) return
  const capturedSamples = buf.chunks.reduce((s, c) => s + c.length, 0)
  if (utteranceDuration < MIN_UTTERANCE_MS || capturedSamples < MIN_SPEECH_SAMPLES || buf.peakRms < MIN_PEAK_RMS) {
    console.log(`[vad] ✗ reject uid=${userId} dur=${utteranceDuration}ms samples=${capturedSamples} peak=${buf.peakRms.toFixed(4)}`)
    buf.chunks = []
    buf.capturing = false
    buf.peakRms = 0
    return
  }

  const chunks = buf.chunks
  const peak = buf.peakRms
  buf.chunks = []
  buf.capturing = false
  buf.peakRms = 0

  if (botSpeaking) {
    console.log(`[vad] ⚡ INTERRUPT uid=${userId} dur=${utteranceDuration}ms peak=${peak.toFixed(4)} — flushing bot audio, aborting pipeline`)
    _botSpeakingUntil = 0
    flushAudio()
    if (_currentAbort) { _currentAbort.abort(); _currentAbort = null }
    buf.processing = true
    handleUtterance(userId, chunks, peak)
    return
  }

  console.log(`[vad] flush uid=${userId} dur=${utteranceDuration}ms silence=${silenceDuration}ms peak=${peak.toFixed(4)}`)
  buf.processing = true
  handleUtterance(userId, chunks, peak)
}
