import { transcribe } from './discord-whisper.js'

const SAMPLE_RATE = 48000
const DEBOUNCE_MS = 600
const MIN_RETRANSCRIBE_SAMPLES = SAMPLE_RATE * 0.6

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
    })
  }
  return sessions.get(userId)
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
    s.latestText = text
    s.latestConf = result.confidence
    s.lastTranscribeAt = Date.now()
    s.lastTranscribeSamples = snapshotSamples
    console.log(`[stream] uid=${userId} streaming STT ${Date.now()-t0}ms samples=${snapshotSamples} → "${text.slice(0,60)}"`)
    for (const fn of s.listeners) try { fn(text, result.confidence) } catch {}
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

export async function finalizeAndClear(userId) {
  const s = sessions.get(userId)
  if (!s) return { text: '', confidence: 0 }
  if (s.chunks.length === 0) return { text: s.latestText, confidence: s.latestConf }
  const pcmBuffer = chunksToPcmBuffer(s.chunks)
  const totalSamples = s.totalSamples
  clear(userId)
  try {
    const result = await transcribe(pcmBuffer, SAMPLE_RATE)
    const text = (result.text || '').trim()
    console.log(`[stream] uid=${userId} finalize samples=${totalSamples} → "${text.slice(0,80)}"`)
    return { text, confidence: result.confidence }
  } catch (err) {
    console.error(`[stream] uid=${userId} finalize error:`, err.message)
    return { text: s.latestText, confidence: s.latestConf }
  }
}

export function clear(userId) {
  const s = sessions.get(userId)
  if (!s) return
  s.chunks = []
  s.totalSamples = 0
  s.lastTranscribeSamples = 0
  s.latestText = ''
  s.latestConf = 0
}

export function onPartial(userId, fn) {
  const s = getSession(userId)
  s.listeners.push(fn)
}
