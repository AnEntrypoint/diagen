import { transcribe } from './discord-whisper.js'
import { resampleAudio } from './server-utils.mjs'
import { synthesize } from './omnivoice-tts-bridge.js'
import { generate as generateLLM, isAvailable as isLLMAvailable } from './llm-ollama.js'
import fs from 'fs'

const SAMPLE_RATE_DISCORD = 48000
const SAMPLE_RATE_TTS_FALLBACK = 24000
const MIN_TRANSCRIPT_CHARS = 3

let voiceReferencePath = null
let voiceReferenceText = null
let characterSystemPrompt = null

export function setCharacterCard(card) {
  const d = card.spec === 'chara_card_v2' ? card.data : card
  const parts = [d.name && `Name: ${d.name}`, d.description, d.personality && `Personality: ${d.personality}`, d.scenario && `Scenario: ${d.scenario}`].filter(Boolean)
  characterSystemPrompt = parts.join('\n\n') || null
  console.log(`[processor] character card set: ${d.name || '(unnamed)'} (${characterSystemPrompt?.length || 0} chars)`)
}

export function getCharacterSystemPrompt() { return characterSystemPrompt }

export function setVoiceEmbedding(refAudioPath) {
  voiceReferencePath = refAudioPath
}

function getVoiceReferenceText() {
  if (voiceReferenceText !== null) return voiceReferenceText
  if (!voiceReferencePath) return null
  const sidecar = voiceReferencePath.replace(/\.wav$/i, '.txt')
  if (!fs.existsSync(sidecar)) {
    console.warn(`[processor] no ref text sidecar at ${sidecar} — voice cloning disabled`)
    voiceReferenceText = ''
    return ''
  }
  voiceReferenceText = fs.readFileSync(sidecar, 'utf8').trim()
  console.log(`[processor] voice reference text loaded from ${sidecar} (${voiceReferenceText.length} chars)`)
  return voiceReferenceText
}

export async function processUserAudio(pcmBuffer, sampleRate, userId) {
  if (!pcmBuffer || pcmBuffer.length === 0) throw new Error(`step=input userId=${userId}: empty audio buffer`)
  if (typeof sampleRate !== 'number' || sampleRate < 8000 || sampleRate > 48000) throw new Error(`step=validate userId=${userId}: sampleRate ${sampleRate} out of range`)
  if (!voiceReferencePath) throw new Error(`step=voiceEmbed userId=${userId}: no voice embedding loaded`)

  console.log(`[processor] userId=${userId} step=transcribe start (${pcmBuffer.length} bytes)`)
  const transcription = await transcribe(pcmBuffer, sampleRate)
  const userText = transcription.text?.trim() || ''
  console.log(`[processor] userId=${userId} step=transcribe done: "${userText}" confidence=${transcription.confidence.toFixed(2)}`)

  if (!userText || userText === '[no speech detected]' || userText.length < MIN_TRANSCRIPT_CHARS) {
    console.log(`[processor] userId=${userId} skip: no real speech transcribed`)
    return null
  }

  if (!(await isLLMAvailable())) {
    console.log(`[processor] userId=${userId} skip: Ollama unavailable, cannot converse`)
    return null
  }

  console.log(`[processor] userId=${userId} step=generate start`)
  const responseText = (await generateLLM(userText, characterSystemPrompt || undefined)).trim()
  console.log(`[processor] userId=${userId} step=generate done: "${responseText.slice(0, 80)}"`)
  if (!responseText) return null

  const refText = getVoiceReferenceText()
  console.log(`[processor] userId=${userId} step=synthesize start (voice_clone=${Boolean(refText)})`)
  const { audio: ttsAudio, sampleRate: ttsSampleRate } = await synthesize(responseText, refText ? voiceReferencePath : null, refText || null)
  console.log(`[processor] userId=${userId} step=synthesize done: ${ttsAudio?.length} samples @ ${ttsSampleRate}Hz`)
  if (!ttsAudio || ttsAudio.length === 0) throw new Error(`step=synthesize userId=${userId}: empty synthesis output`)

  const fromRate = ttsSampleRate || SAMPLE_RATE_TTS_FALLBACK
  const resampled = resampleAudio(ttsAudio, fromRate, SAMPLE_RATE_DISCORD)
  if (!resampled || resampled.length === 0) throw new Error(`step=resample userId=${userId}: empty resampled output`)

  console.log(`[processor] userId=${userId} pipeline complete: "${responseText.slice(0, 50)}..." -> ${resampled.length} samples @ ${fromRate}->${SAMPLE_RATE_DISCORD}Hz`)
  return resampled
}

export default { processUserAudio, setVoiceEmbedding }
