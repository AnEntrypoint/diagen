import { pushAudioFrame } from 'dispipe/voice'
import { pushFrame, onPartial, onStable } from './whisper-stream.js'
import * as speakGate from './speak-gate.js'

const SAMPLE_RATE = 48000
const ACTIVE_RMS = Number(process.env.VAD_ACTIVE_RMS || 0.012)
const TARGET_RMS = 0.15
const MAX_GAIN = 25
const MIN_GAIN = 1
const GAIN_ATTACK = 0.25
const GAIN_MIN_RMS = 0.003
const BOT_SPEAK_TAIL_MS = 250

const userBuffers = new Map()
let _processingQueue = null
let _lastError = null
let _botSpeakingUntil = 0
let _usernameResolver = (uid) => `user${String(uid).slice(-4)}`
const _skippedFrames = new Map()

export function setUsernameResolver(fn) { _usernameResolver = fn }

export function init(processingQueue, lastErrorRef) {
  _processingQueue = processingQueue
  _lastError = lastErrorRef
  speakGate.setAudioSink((monoChunk, _text) => {
    const stereo = new Float32Array(monoChunk.length * 2)
    for (let i = 0; i < monoChunk.length; i++) { stereo[i * 2] = monoChunk[i]; stereo[i * 2 + 1] = monoChunk[i] }
    const durMs = (monoChunk.length / SAMPLE_RATE) * 1000
    const base = Math.max(_botSpeakingUntil, Date.now())
    _botSpeakingUntil = base + durMs + BOT_SPEAK_TAIL_MS
    pushAudioFrame(stereo)
  })
  console.log(`[vad] init mode=state-machine activeRms=${ACTIVE_RMS}`)
}

export function getBuffers() { return userBuffers }
export function getActiveSpeakers() {
  const out = []
  for (const [uid, b] of userBuffers.entries()) {
    out.push({ userId: uid, username: _usernameResolver(uid), gain: b.gain, lastActiveAt: b.lastActiveAt || 0, skipped: _skippedFrames.get(uid) || 0 })
  }
  return out
}

function rms(samples) {
  let sum = 0
  for (let i = 0; i < samples.length; i++) sum += samples[i] * samples[i]
  return Math.sqrt(sum / samples.length)
}

function getOrCreateBuffer(userId) {
  if (!userBuffers.has(userId)) {
    userBuffers.set(userId, { gain: 1, lastActiveAt: 0 })
    console.log(`[vad] new buffer for uid=${userId}`)
    const fire = (text, conf) => {
      const username = _usernameResolver(userId)
      speakGate.noteWhisperWord({ userId, username, text })
    }
    onPartial(userId, fire)
    onStable(userId, fire)
  }
  return userBuffers.get(userId)
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

  if (botSpeaking || rawRms < ACTIVE_RMS) {
    _skippedFrames.set(userId, (_skippedFrames.get(userId) || 0) + 1)
    return
  }
  buf.lastActiveAt = now
  pushFrame(userId, f32)
}
