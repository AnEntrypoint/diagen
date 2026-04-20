import { synthesize } from './omnivoice-tts-bridge.js'
import { resampleAudio } from './server-utils.mjs'
import fs from 'fs'

const PREAMBLES = {
  thinking: [
    'Well hold up now, let me think on that for a second.',
    'Hmm, alright, give me just a second here.',
    'Ah, okay okay, let me see what I got for you.',
    'Yeah yeah, hang on, I was just finishing up.',
    'Alright, let me put this one together real quick.',
    'One second now, thinking about that one.',
    'Hmm, good question — lemme chew on that a bit.',
  ],
  acknowledge: [
    'Yeah, I hear you loud and clear on that.',
    'Right, for sure, no worries at all.',
    'Yeah yeah, makes sense to me.',
    'Mhm, I feel you on that one.',
    'Gotcha, that tracks.',
  ],
  laugh: [
    'Ha, that\'s a good one right there.',
    'Hahah, you got me with that one.',
    'Heh, for real now.',
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
