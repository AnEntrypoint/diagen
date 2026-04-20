import { transcribe } from './discord-whisper.js'

const SAMPLE_RATE = 48000
const DEBOUNCE_MS = 300
const MIN_RETRANSCRIBE_SAMPLES = SAMPLE_RATE * 0.3
const STABILITY_MS = 650
const MIN_WORDS_TO_FIRE = 1

const sessions = new Map()

function getSession(userId) {
  if (!sessions.has(userId)) {
    sessions.set(userId, {
      chunks: [],
      totalSamples: 0,
      lastTranscribeAt: 0,
      lastTranscribeSamples: 0,
      inFlight: false,
      latestText: '',
      latestConf: 0,
      listeners: [],
      stableListeners: [],
      stableText: '',
      stableSince: 0,
      lastFiredText: '',
      stabilityTimer: null,
    })
  }
  return sessions.get(userId)
}

function isSentinel(text) {
  if (!text) return true
  const t = text.trim()
  if (t.length === 0) return true
  return t.charAt(0) === '[' && t.charAt(t.length - 1) === ']' && t.indexOf(']') === t.length - 1
}

function wordCount(text) {
  return text.trim().split(/\s+/).filter(Boolean).length
}

function scheduleStabilityCheck(userId) {
  const s = sessions.get(userId)
  if (!s) return
  if (s.stabilityTimer) clearTimeout(s.stabilityTimer)
  const remain = Math.max(50, STABILITY_MS - (Date.now() - s.stableSince))
  s.stabilityTimer = setTimeout(() => fireIfStable(userId), remain)
}

function fireIfStable(userId) {
  const s = sessions.get(userId)
  if (!s) return
  s.stabilityTimer = null
  const text = s.latestText
  if (isSentinel(text)) return
  if (text !== s.stableText) return
  if (Date.now() - s.stableSince < STABILITY_MS) { scheduleStabilityCheck(userId); return }
  if (wordCount(text) < MIN_WORDS_TO_FIRE) return
  if (text === s.lastFiredText) return
  s.lastFiredText = text
  const conf = s.latestConf
  console.log(`[stream] uid=${userId} ⚡ stable "${text.slice(0,80)}" conf=${conf.toFixed(2)} — firing`)
  clear(userId)
  for (const fn of s.stableListeners) try { fn(text, conf) } catch (e) { console.error('[stream] stable listener err:', e.message) }
}

export function pushFrame(userId, f32Frame) {
  const s = getSession(userId)
  s.chunks.push(f32Frame)
  s.totalSamples += f32Frame.length
  maybeRetranscribe(userId)
}

function chunksToPcmBuffer(chunks) {
  const total = chunks.reduce((a, c) => a + c.length, 0)
  const merged = new Float32Array(total)
  let off = 0
  for (const c of chunks) { merged.set(c, off); off += c.length }
  const int16 = new Int16Array(merged.length)
  for (let i = 0; i < merged.length; i++) {
    const v = Math.max(-1, Math.min(1, merged[i]))
    int16[i] = v < 0 ? v * 0x8000 : v * 0x7FFF
  }
  return Buffer.from(int16.buffer)
}

async function maybeRetranscribe(userId) {
  const s = getSession(userId)
  if (s.inFlight) return
  const now = Date.now()
  const newSinceLast = s.totalSamples - s.lastTranscribeSamples
  if (newSinceLast < MIN_RETRANSCRIBE_SAMPLES) return
  if (now - s.lastTranscribeAt < DEBOUNCE_MS) return

  s.inFlight = true
  const snapshotSamples = s.totalSamples
  const pcmBuffer = chunksToPcmBuffer(s.chunks)
  const t0 = Date.now()
  try {
    const result = await transcribe(pcmBuffer, SAMPLE_RATE)
    const text = (result.text || '').trim()
    const prev = s.latestText
    s.latestText = text
    s.latestConf = result.confidence
    s.lastTranscribeAt = Date.now()
    s.lastTranscribeSamples = snapshotSamples
    console.log(`[stream] uid=${userId} streaming STT ${Date.now()-t0}ms samples=${snapshotSamples} → "${text.slice(0,60)}"`)
    if (text !== s.stableText) {
      s.stableText = text
      s.stableSince = Date.now()
    }
    if (!isSentinel(text) && wordCount(text) >= MIN_WORDS_TO_FIRE) {
      scheduleStabilityCheck(userId)
    }
    for (const fn of s.listeners) try { fn(text, result.confidence, prev) } catch {}
  } catch (err) {
    console.error(`[stream] uid=${userId} transcribe error:`, err.message)
  } finally {
    s.inFlight = false
    setTimeout(() => maybeRetranscribe(userId), 50)
  }
}

export function getLatest(userId) {
  const s = sessions.get(userId)
  if (!s) return { text: '', confidence: 0, samples: 0 }
  return { text: s.latestText, confidence: s.latestConf, samples: s.totalSamples }
}

export function clear(userId) {
  const s = sessions.get(userId)
  if (!s) return
  s.chunks = []
  s.totalSamples = 0
  s.lastTranscribeSamples = 0
  s.latestText = ''
  s.latestConf = 0
  s.stableText = ''
  s.stableSince = 0
  if (s.stabilityTimer) { clearTimeout(s.stabilityTimer); s.stabilityTimer = null }
}

export function onPartial(userId, fn) {
  const s = getSession(userId)
  s.listeners.push(fn)
}

export function onStable(userId, fn) {
  const s = getSession(userId)
  s.stableListeners.push(fn)
}
