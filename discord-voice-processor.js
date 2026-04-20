import { transcribe } from './discord-whisper.js'
import { resampleAudio } from './server-utils.mjs'
import { synthesize } from './omnivoice-tts-bridge.js'
import { generate as generateLLM, generateTokens, isAvailable as isLLMAvailable } from './llm-llamacpp.js'
import fs from 'fs'

const SAMPLE_RATE_DISCORD = 48000
const SAMPLE_RATE_TTS_FALLBACK = 24000
const MIN_TRANSCRIPT_CHARS = 2
const NON_SPEECH_RE = /^[\s\[\(\*_`]*(BLANK_AUDIO|no speech detected|laughter|laughs|chuckles|chuckle|music|applause|silence|inaudible|background noise|coughs?|sighs?|breathing|breath|clears throat)[\s\]\)\*_`\.,!\?-]*$/i

function isNonSpeech(text) {
  if (!text) return true
  const stripped = text.replace(/[\[\(].*?[\]\)]/g, '').replace(/\*.*?\*/g, '').trim()
  if (stripped.length < MIN_TRANSCRIPT_CHARS) return true
  return NON_SPEECH_RE.test(text.trim())
}

let voiceReferencePath = null
let voiceReferenceText = null
let characterSystemPrompt = null
let characterName = 'assistant'

const recentHistory = []
const MAX_HISTORY = 3

let _pendingContinuation = null
let _currentSpeech = null
let _preSynthPromise = null

export function noteBotSpeech(text, durationMs) {
  _currentSpeech = { text, startedAt: Date.now(), durationMs }
  _pendingContinuation = null
  _preSynthPromise = null
}

export function noteBotInterrupted() {
  if (!_currentSpeech) return null
  const elapsed = Date.now() - _currentSpeech.startedAt
  const spokenFraction = Math.max(0, Math.min(1, elapsed / _currentSpeech.durationMs))
  const words = _currentSpeech.text.split(/\s+/).filter(Boolean)
  const spokenWords = Math.floor(words.length * spokenFraction)
  const unsaid = words.slice(spokenWords).join(' ').trim()
  _currentSpeech = null
  if (!unsaid || unsaid.length < 4) { _pendingContinuation = null; return null }
  _pendingContinuation = unsaid
  console.log(`[processor] 🔖 interrupted at ${(spokenFraction*100).toFixed(0)}% — remainder="${unsaid.slice(0,60)}"`)
  return unsaid
}

export function getPendingContinuation() { return _pendingContinuation }
export function clearPendingContinuation() { _pendingContinuation = null; _preSynthPromise = null }

const SEGUE_PREFIXES = ['Anyway, as I was sayin — ', 'Like I was sayin — ', 'But yeah, — ', 'Oye so anyway — ']

export function startPreSynth() {
  if (!_pendingContinuation || _preSynthPromise) return
  const prefix = SEGUE_PREFIXES[Math.floor(Math.random() * SEGUE_PREFIXES.length)]
  const text = prefix + _pendingContinuation
  console.log(`[processor] 🎙️ pre-synth segue: "${text.slice(0,60)}"`)
  const refText = getVoiceReferenceText()
  _preSynthPromise = (async () => {
    const t0 = Date.now()
    try {
      const { audio, sampleRate: sr } = await synthesize(text, refText ? voiceReferencePath : null, refText || null)
      console.log(`[processor] ✓ pre-synth ready ${Date.now()-t0}ms samples=${audio?.length}`)
      return { audio, sampleRate: sr, text }
    } catch (err) {
      console.error(`[processor] pre-synth failed:`, err.message)
      return null
    }
  })()
}

export async function consumePreSynth() {
  if (!_preSynthPromise) return null
  const result = await _preSynthPromise
  _preSynthPromise = null
  _pendingContinuation = null
  if (!result) return null
  const fromRate = result.sampleRate || SAMPLE_RATE_TTS_FALLBACK
  const resampled = resampleAudio(result.audio, fromRate, SAMPLE_RATE_DISCORD)
  return { audio: resampled, text: result.text }
}

export function setCharacterCard(card) {
  const d = card.spec === 'chara_card_v2' ? card.data : card
  const name = d.name || 'the character'
  characterName = name
  const essence = [d.description, d.personality].filter(Boolean).join(' ')
  const scene = d.scenario ? ` Setting: ${d.scenario}` : ''
  characterSystemPrompt = `You are generating the next line in a live voice-chat log. The person speaking is ${name}. ${essence}\n\nEach log line is: name, colon, the words that person said out loud, newline. No narration, no stage directions, no parentheses, no quotes around the line, no translations. Just the spoken words in English — the way people actually talk.\n\nWhen it is ${name}'s turn, write one natural reply — usually two or three sentences, a real conversational beat, not a one-word answer. Then stop. Example format:\n\nJordan: Hey man, you doing okay out here?\n${name}: Yeah boy, been a slow morning. Just watching the pumps and thinking about lunch. What you need?\nJordan: Got any jerky?\n${name}: Aisle two, mira, right by the sodas. The spicy one go quick so grab it if you see it.`
  console.log(`[processor] ✓ card loaded: ${name} | prompt=${characterSystemPrompt.length}ch`)
}

export function getCharacterSystemPrompt() { return characterSystemPrompt }

export function setVoiceEmbedding(refAudioPath) {
  voiceReferencePath = refAudioPath
  console.log(`[processor] voice ref: ${refAudioPath}`)
}

function getVoiceReferenceText() {
  if (voiceReferenceText !== null) return voiceReferenceText
  if (!voiceReferencePath) return null
  const sidecar = voiceReferencePath.replace(/\.wav$/i, '.txt')
  if (!fs.existsSync(sidecar)) {
    console.warn(`[processor] ⚠ no ref-text sidecar ${sidecar} — voice clone DISABLED`)
    voiceReferenceText = ''
    return ''
  }
  voiceReferenceText = fs.readFileSync(sidecar, 'utf8').trim()
  console.log(`[processor] ref-text loaded (${voiceReferenceText.length}ch)`)
  return voiceReferenceText
}

function buildPromptWithHistory(userText, username, preambleText = null) {
  const hist = recentHistory.slice(-MAX_HISTORY * 2)
  const ctx = hist.map(h => `${h.role === 'user' ? (h.username || 'Customer') : characterName}: ${h.text}`).join('\n')
  const tail = preambleText ? `${characterName}: ${preambleText} ` : `${characterName}:`
  return `${ctx ? ctx + '\n' : ''}${username}: ${userText}\n${tail}`
}

export async function processUserAudio(pcmBuffer, sampleRate, userId, signal, username = null) {
  const t0 = Date.now()
  const speaker = username || `user${String(userId).slice(-4)}`
  const tag = `uid=${userId} (${speaker})`
  if (!pcmBuffer || pcmBuffer.length === 0) throw new Error(`step=input ${tag}: empty buffer`)
  if (typeof sampleRate !== 'number' || sampleRate < 8000 || sampleRate > 48000) throw new Error(`step=validate ${tag}: bad sampleRate=${sampleRate}`)
  if (!voiceReferencePath) throw new Error(`step=voiceEmbed ${tag}: no voice ref`)

  console.log(`[pipe] ${tag} ▶ transcribe ${(pcmBuffer.length/1024).toFixed(1)}KB @${sampleRate}Hz`)
  const tTrans = Date.now()
  const transcription = await transcribe(pcmBuffer, sampleRate)
  let userText = transcription.text?.trim() || ''
  userText = userText.replace(/^>+\s*/, '').replace(/\s+/g, ' ').trim()
  const phrases = userText.split(/[,.!?]\s+/).filter(Boolean)
  if (phrases.length >= 5) {
    const counts = new Map()
    for (const p of phrases) counts.set(p, (counts.get(p) || 0) + 1)
    const maxCount = Math.max(...counts.values())
    if (maxCount >= 5) {
      const mostCommon = [...counts.entries()].find(([, c]) => c === maxCount)[0]
      console.log(`[processor] ⚠ whisper hallucination loop detected ("${mostCommon}" x${maxCount}) — collapsing`)
      userText = mostCommon + '.'
    }
  }
  const transMs = Date.now() - tTrans
  console.log(`[pipe] ${tag} ✓ transcribe ${transMs}ms conf=${transcription.confidence.toFixed(2)} → "${userText}"`)

  if (signal?.aborted) { console.log(`[pipe] ${tag} ✗ aborted after transcribe`); return null }
  if (isNonSpeech(userText)) {
    console.log(`[pipe] ${tag} ✗ skip: non-speech "${userText}"`)
    return null
  }

  if (!(await isLLMAvailable())) { console.log(`[pipe] ${tag} ✗ LLM unavailable`); return null }

  const prompt = buildPromptWithHistory(userText, speaker)
  console.log(`[pipe] ${tag} ▶ generate hist=${recentHistory.length} prompt=${prompt.length}ch`)
  const tGen = Date.now()
  let responseText = (await generateLLM(prompt, characterSystemPrompt || undefined, signal))
  const nextTurn = responseText.search(/(?:^|\n|\s)[_*`]{0,2}[A-Za-z][\w`.\- ]{0,24}[_*`]{0,2}\s*:\s/)
  if (nextTurn >= 0) responseText = responseText.slice(0, nextTurn)
  responseText = responseText.trim()
  const genMs = Date.now() - tGen
  console.log(`[pipe] ${tag} ✓ generate ${genMs}ms → "${responseText}"`)
  if (!responseText) return null
  if (signal?.aborted) { console.log(`[pipe] ${tag} ✗ aborted after generate`); return null }

  recentHistory.push({ role: 'user', text: userText, username: speaker })
  recentHistory.push({ role: 'assistant', text: responseText })
  if (recentHistory.length > MAX_HISTORY * 2) recentHistory.splice(0, recentHistory.length - MAX_HISTORY * 2)

  const refText = getVoiceReferenceText()
  console.log(`[pipe] ${tag} ▶ synthesize clone=${Boolean(refText)}`)
  const tTts = Date.now()
  const { audio: ttsAudio, sampleRate: ttsSampleRate } = await synthesize(responseText, refText ? voiceReferencePath : null, refText || null)
  const ttsMs = Date.now() - tTts
  console.log(`[pipe] ${tag} ✓ synthesize ${ttsMs}ms ${ttsAudio?.length} samples @${ttsSampleRate}Hz`)
  if (!ttsAudio || ttsAudio.length === 0) throw new Error(`step=synthesize ${tag}: empty output`)
  if (signal?.aborted) { console.log(`[pipe] ${tag} ✗ aborted after synthesize`); return null }

  const fromRate = ttsSampleRate || SAMPLE_RATE_TTS_FALLBACK
  const resampled = resampleAudio(ttsAudio, fromRate, SAMPLE_RATE_DISCORD)
  if (!resampled || resampled.length === 0) throw new Error(`step=resample ${tag}: empty`)

  const totalMs = Date.now() - t0
  console.log(`[pipe] ${tag} ✅ complete ${totalMs}ms (stt=${transMs} llm=${genMs} tts=${ttsMs}) → ${resampled.length} samples`)
  return resampled
}

function splitSentences(text) {
  const parts = text.match(/[^.!?]+[.!?]+(\s+|$)|[^.!?]+$/g)
  if (!parts) return [text]
  const out = []
  let buf = ''
  for (const p of parts) {
    buf += p
    if (buf.length >= 25 || /[.!?]\s*$/.test(buf)) { out.push(buf.trim()); buf = '' }
  }
  if (buf.trim()) out.push(buf.trim())
  return out.filter(Boolean)
}

export async function processTranscript(rawText, confidence, userId, signal, username = null, onChunk = null, preambleText = null) {
  const t0 = Date.now()
  const speaker = username || `user${String(userId).slice(-4)}`
  const tag = `uid=${userId} (${speaker})`
  if (!voiceReferencePath) throw new Error(`step=voiceEmbed ${tag}: no voice ref`)

  let userText = (rawText || '').trim().replace(/^>+\s*/, '').replace(/\s+/g, ' ').trim()
  const phrases = userText.split(/[,.!?]\s+/).filter(Boolean)
  if (phrases.length >= 5) {
    const counts = new Map()
    for (const p of phrases) counts.set(p, (counts.get(p) || 0) + 1)
    const maxCount = Math.max(...counts.values())
    if (maxCount >= 5) {
      const mostCommon = [...counts.entries()].find(([, c]) => c === maxCount)[0]
      console.log(`[processor] ⚠ whisper hallucination loop ("${mostCommon}" x${maxCount}) — collapsing`)
      userText = mostCommon + '.'
    }
  }
  console.log(`[pipe] ${tag} text="${userText}" conf=${(confidence||0).toFixed(2)}`)

  if (signal?.aborted) return null
  if (isNonSpeech(userText)) {
    console.log(`[pipe] ${tag} ✗ skip: non-speech "${userText}"`)
    return null
  }
  if (!(await isLLMAvailable())) { console.log(`[pipe] ${tag} ✗ LLM unavailable`); return null }

  const prompt = buildPromptWithHistory(userText, speaker, preambleText)
  console.log(`[pipe] ${tag} ▶ stream-gen hist=${recentHistory.length}${preambleText ? ` preamble="${preambleText}"` : ''}`)
  const refText = getVoiceReferenceText()
  const tGen = Date.now()

  let accum = ''
  let responseText = ''
  let ttsChain = Promise.resolve()
  const allAudio = []
  let firstTokenAt = null
  let firstAudioAt = null
  let stopRequested = false

  const flushSentence = (sent) => {
    const piece = sent.trim()
    if (!piece) return
    responseText += (responseText ? ' ' : '') + piece
    ttsChain = ttsChain.then(async () => {
      if (signal?.aborted || stopRequested) return
      const tc = Date.now()
      try {
        const { audio, sampleRate: sr } = await synthesize(piece, refText ? voiceReferencePath : null, refText || null)
        const resampled = resampleAudio(audio, sr || SAMPLE_RATE_TTS_FALLBACK, SAMPLE_RATE_DISCORD)
        if (signal?.aborted || stopRequested) return
        if (!firstAudioAt) { firstAudioAt = Date.now(); console.log(`[pipe] ${tag} 🎵 first-audio ${firstAudioAt - t0}ms`) }
        console.log(`[pipe] ${tag} ✓ tts ${Date.now()-tc}ms "${piece.slice(0,40)}"`)
        if (onChunk) onChunk(resampled)
        allAudio.push(resampled)
      } catch (err) {
        console.error(`[pipe] ${tag} tts error:`, err.message)
      }
    })
  }

  const MAX_RESPONSE_CHARS = 280
  try {
    for await (const tok of generateTokens(prompt, characterSystemPrompt || undefined, signal)) {
      if (!firstTokenAt) { firstTokenAt = Date.now(); console.log(`[pipe] ${tag} ⚡ first-token ${firstTokenAt - t0}ms`) }
      accum += tok
      const turnIdx = accum.search(/(?:^|\n|\s)[_*`]{0,2}[A-Za-z][\w`.\- ]{0,24}[_*`]{0,2}\s*:\s/)
      if (turnIdx >= 0) { accum = accum.slice(0, turnIdx); stopRequested = true; break }
      if (responseText.length + accum.length > MAX_RESPONSE_CHARS) { stopRequested = true; break }
      const isFirst = !firstAudioAt && allAudio.length === 0 && ttsChain === Promise.resolve ? true : (allAudio.length === 0 && !firstAudioAt)
      const splitRe = isFirst ? /^([\s\S]*?[,;:.!?])(\s+|$)/ : /^([\s\S]*?[.!?])(\s+|$)/
      const minLen = isFirst ? 12 : 20
      let m
      let pending = ''
      while ((m = accum.match(splitRe))) {
        pending += m[1] + ' '
        accum = accum.slice(m[0].length)
        if (pending.length >= minLen) { flushSentence(pending.trim()); pending = '' }
        if (responseText.length > MAX_RESPONSE_CHARS) { stopRequested = true; break }
      }
      if (pending.trim()) accum = pending + accum
      if (stopRequested) break
    }
  } catch (err) {
    if (err.name !== 'AbortError') throw err
  }
  if (!stopRequested && accum.trim()) flushSentence(accum)
  await ttsChain

  responseText = responseText.trim()
  const genMs = Date.now() - tGen
  console.log(`[pipe] ${tag} ✓ stream-gen total ${genMs}ms → "${responseText}"`)
  if (!responseText || signal?.aborted) return null

  recentHistory.push({ role: 'user', text: userText, username: speaker })
  recentHistory.push({ role: 'assistant', text: responseText })
  if (recentHistory.length > MAX_HISTORY * 2) recentHistory.splice(0, recentHistory.length - MAX_HISTORY * 2)

  const totalLen = allAudio.reduce((s, a) => s + a.length, 0)
  const combined = new Float32Array(totalLen)
  let off = 0
  for (const a of allAudio) { combined.set(a, off); off += a.length }
  const totalMs = Date.now() - t0
  console.log(`[pipe] ${tag} ✅ complete ${totalMs}ms first-audio=${firstAudioAt ? firstAudioAt - t0 : '?'}ms chunks=${allAudio.length}`)
  combined._text = responseText
  return combined
}

const _speculativeCache = new Map()

export function startSpeculativeGenerate(userId, userText, username) {
  if (!userText || userText.length < 8) return
  const cached = _speculativeCache.get(userId)
  if (cached && cached.userText === userText) return
  if (cached && cached.abort) cached.abort.abort()
  const speaker = username || `user${String(userId).slice(-4)}`
  const prompt = buildPromptWithHistory(userText, speaker)
  const abort = new AbortController()
  const resultPromise = generateLLM(prompt, characterSystemPrompt || undefined, abort.signal).catch(() => null)
  _speculativeCache.set(userId, { userText, abort, resultPromise })
  console.log(`[processor] 🔮 speculative LLM uid=${userId} on "${userText.slice(0,40)}"`)
}

export function cancelSpeculative(userId) {
  const c = _speculativeCache.get(userId)
  if (c?.abort) c.abort.abort()
  _speculativeCache.delete(userId)
}

export function clearHistory() { recentHistory.length = 0; console.log('[processor] history cleared') }

export default { processUserAudio, processTranscript, setVoiceEmbedding, setCharacterCard, clearHistory }
