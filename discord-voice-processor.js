import { transcribe } from './discord-whisper.js'
import { resampleAudio } from './server-utils.mjs'
import { synthesize } from './omnivoice-tts-bridge.js'
import { generate as generateLLM, isAvailable as isLLMAvailable } from './llm-ollama.js'

const SAMPLE_RATE_DISCORD = 48000
const SAMPLE_RATE_TTS = 24000

let voiceReferencePath = null

export function setVoiceEmbedding(refAudioPath) {
  voiceReferencePath = refAudioPath
}

async function generateResponse(text, userId) {
  if (!text || text === '[no speech detected]') return "I didn't catch that. Could you speak again?"
  const available = await isLLMAvailable()
  if (!available) return `You said: "${text.slice(0, 100)}"`
  return generateLLM(text)
}

export async function processUserAudio(pcmBuffer, sampleRate, userId) {
  if (!pcmBuffer || pcmBuffer.length === 0) throw new Error(`step=input userId=${userId}: empty audio buffer`)
  if (typeof sampleRate !== 'number' || sampleRate < 8000 || sampleRate > 48000) throw new Error(`step=validate userId=${userId}: sampleRate ${sampleRate} out of range`)
  if (!voiceReferencePath) throw new Error(`step=voiceEmbed userId=${userId}: no voice embedding loaded`)

  try {
    const transcription = await transcribe(pcmBuffer, sampleRate)
    if (transcription.confidence < 0.01) console.log(`[processor] userId=${userId} low confidence: ${transcription.confidence}`)

    const responseText = await generateResponse(transcription.text, userId)

    const ttsOutput = await synthesize(responseText, voiceReferencePath, 'reference speech')
    if (!ttsOutput || ttsOutput.length === 0) throw new Error(`step=synthesize userId=${userId}: empty synthesis output`)

    const resampled = resampleAudio(ttsOutput, SAMPLE_RATE_TTS, SAMPLE_RATE_DISCORD)
    if (!resampled || resampled.length === 0) throw new Error(`step=resample userId=${userId}: empty resampled output`)

    console.log(`[processor] userId=${userId} pipeline complete: "${responseText.slice(0, 50)}..." -> ${resampled.length} samples`)
    return resampled
  } catch (err) {
    throw err.message.includes('step=') ? err : new Error(`step=pipeline userId=${userId}: ${err.message}`)
  }
}

export default { processUserAudio, setVoiceEmbedding }
