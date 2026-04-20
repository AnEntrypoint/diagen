import { processTranscript, noteBotSpeech, noteBotInterrupted, startSpeculativeGenerate } from './discord-voice-processor.js'
import { pushAudioFrame, flushAudio } from 'dispipe/voice'
import { pushFrame, onStable, onPartial } from './whisper-stream.js'
import { pick as pickPreamble, isReady as preambleReady } from './preamble-cache.js'

const SAMPLE_RATE = 48000
const INTERRUPT_THRESHOLD = 0.05
const INTERRUPT_SUSTAIN_MS = 250
const BOT_SPEAK_TAIL_MS = 250
const TARGET_RMS = 0.08
const MAX_GAIN = 15
const MIN_GAIN = 1
const GAIN_ATTACK = 0.15
const GAIN_MIN_RMS = 0.004

const userBuffers = new Map()
let _processingQueue = null
let _lastError = null
let _botSpeakingUntil = 0
let _currentAbort = null
let _currentUserId = null
let _activeUtteranceCount = 0
let _usernameResolver = (uid) => `user${String(uid).slice(-4)}`

export function setUsernameResolver(fn) { _usernameResolver = fn }

export function init(processingQueue, lastErrorRef) {
  _processingQueue = processingQueue
  _lastError = lastErrorRef
  console.log(`[vad] init mode=continuous-stream interrupt=${INTERRUPT_THRESHOLD}`)
}

export function getBuffers() { return userBuffers }

function rms(samples) {
  let sum = 0
  for (let i = 0; i < samples.length; i++) sum += samples[i] * samples[i]
  return Math.sqrt(sum / samples.length)
}

function getOrCreateBuffer(userId) {
  if (!userBuffers.has(userId)) {
    userBuffers.set(userId, { gain: 1, loudSince: 0 })
    console.log(`[vad] new buffer for uid=${userId}`)
    onPartial(userId, (text, conf) => {
      if (conf > 0.15 && text.length >= 8) {
        const username = _usernameResolver(userId)
        startSpeculativeGenerate(userId, text, username)
      }
    })
    onStable(userId, (text, conf) => {
      handleUtterance(userId, text, conf).catch(err => console.error('[vad] handleUtterance threw:', err.stack || err.message))
    })
  }
  return userBuffers.get(userId)
}

async function handleUtterance(userId, text, confidence) {
  if (_currentAbort) {
    console.log(`[vad] ⏸ pre-empting prior utterance (was uid=${_currentUserId}) for new uid=${userId}`)
    try { _currentAbort.abort() } catch {}
    _currentAbort = null
  }
  _activeUtteranceCount++
  console.log(`[vad] ▶ utterance uid=${userId} text="${text.slice(0,60)}" conf=${confidence.toFixed(2)} active=${_activeUtteranceCount}`)

  const abort = new AbortController()
  _currentAbort = abort
  _currentUserId = userId
  const entry = { userId, startTime: Date.now() }
  _processingQueue.push(entry)

  let preambleStaged = preambleReady() ? pickPreamble('thinking') : null
  try {
    const username = _usernameResolver(userId)
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
        console.log(`[vad] ⚡ preamble "${pre.text}" (${preDurMs.toFixed(0)}ms)`)
      }
      const stereo = new Float32Array(monoChunk.length * 2)
      for (let i = 0; i < monoChunk.length; i++) { stereo[i * 2] = monoChunk[i]; stereo[i * 2 + 1] = monoChunk[i] }
      const durMs = (monoChunk.length / SAMPLE_RATE) * 1000
      const base = Math.max(_botSpeakingUntil, Date.now())
      _botSpeakingUntil = base + durMs + BOT_SPEAK_TAIL_MS
      pushAudioFrame(stereo)
      if (!firstChunkAt) { firstChunkAt = Date.now(); console.log(`[vad] 🎵 first-chunk uid=${userId} TTFA=${firstChunkAt-entry.startTime}ms`) }
    }
    const preambleHintText = preambleStaged ? preambleStaged.text : null
    const monoOut = await processTranscript(text, confidence, userId, abort.signal, username, onChunk, preambleHintText)
    if (!monoOut || abort.signal.aborted) { console.log(`[vad] uid=${userId} no output (aborted=${abort.signal.aborted})`); return }
    const totalDurMs = (monoOut.length / SAMPLE_RATE) * 1000
    noteBotSpeech(monoOut._text || '[response]', totalDurMs)
    console.log(`[vad] 📢 speaking-total uid=${userId} ${totalDurMs.toFixed(0)}ms`)
  } catch (err) {
    if (err.name === 'AbortError' || err.message?.includes('aborted')) {
      console.log(`[vad] uid=${userId} pipeline aborted cleanly`)
    } else {
      console.error(`[vad] ✗ uid=${userId} pipeline error:`, err.stack || err.message)
      _lastError.value = { message: err.message, timestamp: Date.now(), userId }
    }
  } finally {
    if (_currentAbort === abort) { _currentAbort = null; _currentUserId = null }
    const idx = _processingQueue.indexOf(entry)
    if (idx !== -1) _processingQueue.splice(idx, 1)
    _activeUtteranceCount--
    const elapsed = Date.now() - entry.startTime
    console.log(`[vad] ◀ done uid=${userId} totalMs=${elapsed} active=${_activeUtteranceCount}`)
  }
}

export function onPcmChunk(userId, stereoF32) {
  const now = Date.now()
  const botSpeaking = now < _botSpeakingUntil
  const monoLen = stereoF32.length / 2
  const raw = new Float32Array(monoLen)
  for (let i = 0; i < monoLen; i++) raw[i] = (stereoF32[i * 2] + stereoF32[i * 2 + 1]) * 0.5

  const buf = getOrCreateBuffer(userId)
  const rawRms = rms(raw)

  if (rawRms > GAIN_MIN_RMS) {
    const wantGain = Math.max(MIN_GAIN, Math.min(MAX_GAIN, TARGET_RMS / rawRms))
    buf.gain = buf.gain * (1 - GAIN_ATTACK) + wantGain * GAIN_ATTACK
  }
  const g = buf.gain
  const f32 = new Float32Array(monoLen)
  for (let i = 0; i < monoLen; i++) {
    const v = raw[i] * g
    f32[i] = v > 1 ? 1 : v < -1 ? -1 : v
  }

  pushFrame(userId, f32)

  if (botSpeaking && rawRms > INTERRUPT_THRESHOLD) {
    if (!buf.loudSince) buf.loudSince = now
    if (now - buf.loudSince >= INTERRUPT_SUSTAIN_MS) {
      console.log(`[vad] ⚡ INTERRUPT uid=${userId} rms=${rawRms.toFixed(4)} sustained ${now - buf.loudSince}ms`)
      noteBotInterrupted()
      _botSpeakingUntil = 0
      flushAudio()
      if (_currentAbort) { try { _currentAbort.abort() } catch {}; _currentAbort = null }
      buf.loudSince = 0
    }
  } else {
    buf.loudSince = 0
  }
}
