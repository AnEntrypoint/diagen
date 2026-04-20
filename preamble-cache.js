import { synthesize } from './omnivoice-tts-bridge.js'
import { resampleAudio } from './server-utils.mjs'
import fs from 'fs'

const PREAMBLES = {
  thinking: [
    'Mira, let me think on that for a second boy.',
    'Oye, hold up, I gotta figure this one out.',
    'Hmm, yeah, give me a second here amigo.',
    'Ah, okay okay, let me see what I got for you.',
    'Eh, alright, let me put this together real quick.',
    'Yeah yeah, hang on, I was just finishing up here.',
    'Oye boy, one second, thinking about that one.',
  ],
  acknowledge: [
    'Yeah boy, I hear you loud and clear.',
    'Mira, sí sí, I got you on that one.',
    'Oye, for sure amigo, no worries at all.',
    'Right right, yeah, makes sense to me.',
    'Sí sí boy, I feel you on that.',
  ],
  laugh: [
    'Ha, oye, that\'s a good one boy.',
    'Hahah, mira, you got me with that.',
    'Eh eh eh, amigo, for real now.',
  ],
}

const cache = new Map()
let _refPath = null
let _refText = null
let _ready = false

export function setRef(refAudioPath, refText) {
  _refPath = refAudioPath
  _refText = refText
}

export async function warmup(refAudioPath, refText) {
  setRef(refAudioPath, refText)
  const all = Object.entries(PREAMBLES).flatMap(([cat, arr]) => arr.map(text => ({ cat, text })))
  console.log(`[preamble] warming ${all.length} preambles...`)
  const t0 = Date.now()
  for (const { cat, text } of all) {
    try {
      const { audio, sampleRate } = await synthesize(text, _refPath, _refText)
      const resampled = resampleAudio(audio, sampleRate || 24000, 48000)
      cache.set(text, { cat, text, audio: resampled })
    } catch (err) {
      console.warn(`[preamble] failed "${text}":`, err.message)
    }
  }
  _ready = true
  console.log(`[preamble] ✓ ready ${cache.size}/${all.length} in ${((Date.now()-t0)/1000).toFixed(1)}s`)
}

export function isReady() { return _ready }

export function pick(category = 'thinking') {
  if (!_ready || cache.size === 0) return null
  const candidates = [...cache.values()].filter(p => p.cat === category)
  if (candidates.length === 0) return null
  return candidates[Math.floor(Math.random() * candidates.length)]
}

export function stats() {
  const byCat = {}
  for (const p of cache.values()) byCat[p.cat] = (byCat[p.cat] || 0) + 1
  return { ready: _ready, count: cache.size, byCat }
}
