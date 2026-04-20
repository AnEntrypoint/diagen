import { processUserAudio, processTranscript, noteBotSpeech, noteBotInterrupted, startPreSynth, consumePreSynth, clearPendingContinuation, startSpeculativeGenerate, cancelSpeculative } from './discord-voice-processor.js'
import { pushAudioFrame, flushAudio } from 'dispipe/voice'
import { pushFrame, finalizeAndClear, clear as clearStream, onPartial } from './whisper-stream.js'
import { pick as pickPreamble, isReady as preambleReady } from './preamble-cache.js'

const SILENCE_THRESHOLD_BASE = 0.004
const INTERRUPT_THRESHOLD = 0.025
const SILENCE_DURATION_MS = 380
const INTERRUPT_SILENCE_MS = 280
const MIN_UTTERANCE_MS = 350
const MIN_SPEECH_SAMPLES = 48000 * 0.4
const MAX_UTTERANCE_MS = 15000
const SAMPLE_RATE = 48000
const BOT_SPEAK_TAIL_MS = 250
const PREROLL_MS = 500
const PREROLL_SAMPLES = SAMPLE_RATE * PREROLL_MS / 1000
const MIN_PEAK_RMS = 0.020

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
    onPartial(userId, (text, conf, prev) => {
      if (conf > 0.15 && text.length >= 8) {
        const username = _usernameResolver(userId)
        startSpeculativeGenerate(userId, text, username)
      }
    })
  }
  return userBuffers.get(userId)
}

function rms(samples) {
  let sum = 0
  for (let i = 0; i < samples.length; i++) sum += samples[i] * samples[i]
  return Math.sqrt(sum / samples.length)
}

async function handleUtterance(userId, utteranceDurMs, peakRms) {
  if (_currentAbort) {
    console.log(`[vad] ⏸ pre-empting prior utterance for new uid=${userId}`)
    try { _currentAbort.abort() } catch {}
    _currentAbort = null
  }
  _activeUtteranceCount++
  console.log(`[vad] ▶ utterance uid=${userId} dur=${(utteranceDurMs/1000).toFixed(2)}s peak=${peakRms.toFixed(4)} active=${_activeUtteranceCount}`)

  const abort = new AbortController()
  _currentAbort = abort
  const entry = { userId, startTime: Date.now() }
  _processingQueue.push(entry)

  let preambleStaged = null
  try {
    const { text, confidence } = await finalizeAndClear(userId)
    if (abort.signal.aborted) { console.log(`[vad] uid=${userId} aborted before generate`); return }
    const username = _usernameResolver(userId)

    const cleanText = (text || '').trim()
    const isUsable = cleanText && cleanText !== '[no speech detected]' && cleanText.length >= 2
    if (isUsable && preambleReady()) {
      preambleStaged = pickPreamble('thinking')
    }
    let firstChunkAt = null
    const onChunk = (monoChunk) => {
      if (abort.signal.aborted) return
      if (preambleStaged) {
        const pre = preambleStaged
        preambleStaged = null
        const preStereo = new Float32Array(pre.audio.length * 2)
        for (let i = 0; i < pre.audio.length; i++) { preStereo[i * 2] = pre.audio[i]; preStereo[i * 2 + 1] = pre.audio[i] }
        const preDurMs = (pre.audio.length / SAMPLE_RATE) * 1000
        _botSpeakingUntil = Date.now() + preDurMs + BOT_SPEAK_TAIL_MS
        pushAudioFrame(preStereo)
        console.log(`[vad] ⚡ preamble "${pre.text}" (${preDurMs.toFixed(0)}ms) — attached to real response`)
      }
      const stereo = new Float32Array(monoChunk.length * 2)
      for (let i = 0; i < monoChunk.length; i++) { stereo[i * 2] = monoChunk[i]; stereo[i * 2 + 1] = monoChunk[i] }
      const durMs = (monoChunk.length / SAMPLE_RATE) * 1000
      const base = Math.max(_botSpeakingUntil, Date.now())
      _botSpeakingUntil = base + durMs + BOT_SPEAK_TAIL_MS
      pushAudioFrame(stereo)
      if (!firstChunkAt) { firstChunkAt = Date.now(); console.log(`[vad] 🎵 first-chunk uid=${userId} TTFA=${firstChunkAt-entry.startTime}ms dur=${durMs.toFixed(0)}ms`); clearStream(userId) }
    }
    const preambleHintText = preambleStaged ? preambleStaged.text : null
    const monoOut = await processTranscript(text, confidence, userId, abort.signal, username, onChunk, preambleHintText)
    if (!monoOut || abort.signal.aborted) {
      console.log(`[vad] uid=${userId} no output (aborted=${abort.signal.aborted})`)
      return
    }
    const totalDurMs = (monoOut.length / SAMPLE_RATE) * 1000
    noteBotSpeech(monoOut._text || '[response]', totalDurMs)
    console.log(`[vad] 📢 speaking-total uid=${userId} ${totalDurMs.toFixed(0)}ms → until ${new Date(_botSpeakingUntil).toISOString().slice(14,23)}`)

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

  const level = rms(f32)
  const effectiveThreshold = botSpeaking ? INTERRUPT_THRESHOLD : SILENCE_THRESHOLD_BASE
  const isSpeech = level > effectiveThreshold

  pushFrame(userId, f32)

  buf.preroll.push(f32)
  buf.prerollSamples += f32.length
  while (buf.prerollSamples > PREROLL_SAMPLES && buf.preroll.length > 1) {
    const dropped = buf.preroll.shift()
    buf.prerollSamples -= dropped.length
  }

  if (!buf.capturing) {
    if (!isSpeech) return
    buf.capturing = true
    buf.startTime = now - (buf.prerollSamples / SAMPLE_RATE * 1000)
    buf.lastVoiceTime = now
    buf.peakRms = level
    console.log(`[vad] 🎤 speech-start uid=${userId} rms=${level.toFixed(4)} botSpeaking=${botSpeaking} preroll=${(buf.prerollSamples/SAMPLE_RATE*1000).toFixed(0)}ms`)
  } else {
    if (isSpeech) {
      if (level > buf.peakRms) buf.peakRms = level
      buf.lastVoiceTime = now
    }
  }

  const utteranceDuration = now - buf.startTime
  const silenceDuration = now - buf.lastVoiceTime
  const flushSilence = botSpeaking ? INTERRUPT_SILENCE_MS : SILENCE_DURATION_MS
  const shouldFlush = silenceDuration >= flushSilence || utteranceDuration >= MAX_UTTERANCE_MS

  if (!shouldFlush) return
  if (utteranceDuration < MIN_UTTERANCE_MS || buf.peakRms < MIN_PEAK_RMS) {
    console.log(`[vad] ✗ reject uid=${userId} dur=${utteranceDuration}ms peak=${buf.peakRms.toFixed(4)}`)
    clearStream(userId)
    buf.capturing = false
    buf.peakRms = 0
    buf.preroll = []
    buf.prerollSamples = 0
    return
  }

  const peak = buf.peakRms
  buf.capturing = false
  buf.peakRms = 0
  buf.preroll = []
  buf.prerollSamples = 0

  if (botSpeaking) {
    console.log(`[vad] ⚡ INTERRUPT uid=${userId} dur=${utteranceDuration}ms peak=${peak.toFixed(4)}`)
    noteBotInterrupted()
    _botSpeakingUntil = 0
    flushAudio()
    if (_currentAbort) { _currentAbort.abort(); _currentAbort = null }
    buf.processing = true
    handleUtterance(userId, utteranceDuration, peak)
    return
  }

  console.log(`[vad] flush uid=${userId} dur=${utteranceDuration}ms silence=${silenceDuration}ms peak=${peak.toFixed(4)}`)
  buf.processing = true
  handleUtterance(userId, utteranceDuration, peak)
}
