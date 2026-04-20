import { transcribe } from './discord-whisper.js'
import { resampleAudio } from './server-utils.mjs'
import { synthesize } from './omnivoice-tts-bridge.js'
import { generate as generateLLM, generateTokens, isAvailable as isLLMAvailable } from './llm-llamacpp.js'
import fs from 'fs'

const SAMPLE_RATE_DISCORD = 48000
const SAMPLE_RATE_TTS_FALLBACK = 24000
const MIN_TRANSCRIPT_CHARS = 2

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
  const words = _currentSpeech.text.split(' ').filter(Boolean)
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
  characterSystemPrompt = [
    `You are ${name}. Stay in character. ${essence}`,
    ``,
    `This is a live voice chat. You play ${name}. Reply with ${name}'s next spoken turn — the actual words ${name} would say out loud.`,
    ``,
    `Format rules (all strict):`,
    `- Output only the words ${name} speaks. Nothing else. No labels, no names, no colons.`,
    `- No narration, no actions in parentheses, no asterisks, no brackets, no stage directions.`,
    `- Do not write the other person's line or their name. Just your own turn, then stop.`,
    `- Do not repeat what the other person said. Do not translate their words into another language.`,
    `- Reply in the same language the other person used. English in, English out. Use occasional Spanglish only if the other person did first or the character's own signature words ask for it — don't switch to Spanish unprompted.`,
    ``,
    `Length:`,
    `- Two or three full sentences is the target. A real conversational beat — you greet, you answer, you give a hook for them to come back.`,
    `- Minimum: one complete sentence of at least eight words. Never a single-word reply like "Amigo." or "Mira." alone — that reads as broken.`,
    `- Maximum: about forty words.`,
    ``,
    `Handling bad input:`,
    `- If the message is obviously noise, laughter, coughing, music, or gibberish, respond with a short in-character question that invites them to repeat — for example "eh, didn't catch that, say again?" — still a full sentence, still in character, and stop.`,
    ``,
    `Stop after one reply. One turn only. Never continue the conversation past your line.`,
  ].join('\n')
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
  const lower = voiceReferencePath.toLowerCase()
  const sidecar = lower.endsWith('.wav') ? voiceReferencePath.slice(0, -4) + '.txt' : voiceReferencePath + '.txt'
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
  const turns = []
  for (const h of hist) {
    if (h.role === 'user') turns.push(`Them: ${h.text}`)
    else turns.push(`You: ${h.text}`)
  }
  turns.push(`Them: ${userText}`)
  const tail = preambleText ? `You: ${preambleText} ` : `You:`
  return turns.join('\n') + '\n' + tail
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
  const userText = (transcription.text || '').trim()
  const transMs = Date.now() - tTrans
  console.log(`[pipe] ${tag} ✓ transcribe ${transMs}ms conf=${transcription.confidence.toFixed(2)} → "${userText}"`)

  if (signal?.aborted) { console.log(`[pipe] ${tag} ✗ aborted after transcribe`); return null }
  if (!userText || userText.length < MIN_TRANSCRIPT_CHARS) {
    console.log(`[pipe] ${tag} ✗ skip: empty transcript`)
    return null
  }

  if (!(await isLLMAvailable())) { console.log(`[pipe] ${tag} ✗ LLM unavailable`); return null }

  const prompt = buildPromptWithHistory(userText, speaker)
  console.log(`[pipe] ${tag} ▶ generate hist=${recentHistory.length} prompt=${prompt.length}ch`)
  const tGen = Date.now()
  let responseText = (await generateLLM(prompt, characterSystemPrompt || undefined, signal)).trim()
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

const SENTENCE_END_CHARS = new Set(['.', '!', '?'])
const SOFT_BREAK_CHARS = new Set([',', ';', ':', '—', '–'])
const FIRST_CHUNK_MIN = 14
const NEXT_CHUNK_MIN = 24
const MAX_RESPONSE_CHARS = 280

export async function processTranscript(rawText, confidence, userId, signal, username = null, onChunk = null, preambleText = null) {
  const t0 = Date.now()
  const speaker = username || `user${String(userId).slice(-4)}`
  const tag = `uid=${userId} (${speaker})`
  if (!voiceReferencePath) throw new Error(`step=voiceEmbed ${tag}: no voice ref`)

  const userText = (rawText || '').trim()
  console.log(`[pipe] ${tag} text="${userText}" conf=${(confidence||0).toFixed(2)}`)

  if (signal?.aborted) return null
  if (!userText || userText.length < MIN_TRANSCRIPT_CHARS) {
    console.log(`[pipe] ${tag} ✗ skip: empty transcript`)
    return null
  }
  if (!(await isLLMAvailable())) { console.log(`[pipe] ${tag} ✗ LLM unavailable`); return null }

  const prompt = buildPromptWithHistory(userText, speaker, preambleText)
  console.log(`[pipe] ${tag} ▶ stream-gen hist=${recentHistory.length}${preambleText ? ` preamble="${preambleText}"` : ''}`)
  const refText = getVoiceReferenceText()
  const tGen = Date.now()

  let pending = ''
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

  const tryFlush = () => {
    if (!pending) return
    const minLen = allAudio.length === 0 && !firstAudioAt ? FIRST_CHUNK_MIN : NEXT_CHUNK_MIN
    if (pending.length < minLen) return
    let cut = -1
    for (let i = pending.length - 1; i >= minLen - 1; i--) {
      if (SENTENCE_END_CHARS.has(pending[i]) && (i + 1 === pending.length || pending[i+1] === ' ')) { cut = i + 1; break }
    }
    if (cut < 0) {
      for (let i = pending.length - 1; i >= minLen - 1; i--) {
        if (SOFT_BREAK_CHARS.has(pending[i]) && (i + 1 === pending.length || pending[i+1] === ' ')) { cut = i + 1; break }
      }
    }
    if (cut < 0) return
    flushSentence(pending.slice(0, cut))
    pending = pending.slice(cut).trimStart()
  }

  try {
    for await (const tok of generateTokens(prompt, characterSystemPrompt || undefined, signal)) {
      if (!firstTokenAt) { firstTokenAt = Date.now(); console.log(`[pipe] ${tag} ⚡ first-token ${firstTokenAt - t0}ms`) }
      pending += tok
      if (responseText.length + pending.length > MAX_RESPONSE_CHARS) { stopRequested = true; break }
      tryFlush()
      if (stopRequested) break
    }
  } catch (err) {
    if (err.name !== 'AbortError') throw err
  }
  const tail = pending.trim()
  if (!stopRequested && tail) flushSentence(tail)
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
