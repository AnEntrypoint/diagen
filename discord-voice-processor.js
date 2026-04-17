import { transcribe } from './discord-whisper.js'
import { resampleAudio } from './server-utils.mjs'
import { synthesize } from './omnivoice-tts-bridge.js'
import { generate as generateLLM, isAvailable as isLLMAvailable } from './llm-ollama.js'
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

function buildPromptWithHistory(userText, username) {
  const hist = recentHistory.slice(-MAX_HISTORY * 2)
  const ctx = hist.map(h => `${h.role === 'user' ? (h.username || 'Customer') : characterName}: ${h.text}`).join('\n')
  return `${ctx ? ctx + '\n' : ''}${username}: ${userText}\n${characterName}:`
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
  if (!userText || userText === '[no speech detected]' || userText.length < MIN_TRANSCRIPT_CHARS) {
    console.log(`[pipe] ${tag} ✗ skip: trivial transcript (${userText.length}ch)`)
    return null
  }

  if (!(await isLLMAvailable())) { console.log(`[pipe] ${tag} ✗ Ollama unavailable`); return null }

  const prompt = buildPromptWithHistory(userText, speaker)
  console.log(`[pipe] ${tag} ▶ generate hist=${recentHistory.length} prompt=${prompt.length}ch`)
  const tGen = Date.now()
  let responseText = (await generateLLM(prompt, characterSystemPrompt || undefined, signal))
  const nextTurn = responseText.search(/\n\s*[A-Za-z`][\w`.\-]{0,30}\s*:/)
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

export function clearHistory() { recentHistory.length = 0; console.log('[processor] history cleared') }

export default { processUserAudio, setVoiceEmbedding, setCharacterCard, clearHistory }
