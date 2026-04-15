import { transcribe } from './discord-whisper.js'
import { resampleAudio } from './server-utils.mjs'
import { createRequire } from 'module'

const require = createRequire(import.meta.url)
const ttsOnnx = require('webtalk/server-tts-onnx')

// Configuration
const SAMPLE_RATE_DISCORD = 48000 // Discord voice streams are 48kHz
const SAMPLE_RATE_TTS = 24000    // TTS model outputs 24kHz
const TTS_MODELS_DIR = './models/tts'

let ttsModelsLoaded = false
let globalVoiceEmbedding = null

/**
 * Load TTS models once on first use
 */
async function ensureTtsModels() {
  if (ttsModelsLoaded) return
  try {
    await ttsOnnx.loadModels(TTS_MODELS_DIR)
    ttsModelsLoaded = true
  } catch (err) {
    throw new Error(`Failed to load TTS models: ${err.message}`)
  }
}

/**
 * Set the global voice embedding for synthesis
 * @param {Tensor} embedding - Voice embedding tensor from ttsOnnx.encodeVoiceAudio
 */
export function setVoiceEmbedding(embedding) {
  globalVoiceEmbedding = embedding
}

/**
 * Generate response text from transcribed input
 * Simple template-based handler for now
 * @param {string} text - Transcribed user input
 * @param {string} userId - Discord user ID
 * @returns {string} Response text
 */
function generateResponse(text, userId) {
  // Template-based response handler
  if (!text || text === '[no speech detected]') {
    return "I didn't catch that. Could you speak again?"
  }

  const response = `You said: "${text.slice(0, 100)}". ${
    text.length > 100 ? 'Processing...' : ''
  }`
  return response
}

/**
 * Convert float32 audio to int16 PCM Uint8Array
 * Clamps values to [-1, 1] range and scales to int16
 * @param {Float32Array} float32Data - Audio samples in [-1, 1] range
 * @returns {Uint8Array} 16-bit PCM in little-endian format
 */
function float32ToInt16PCM(float32Data) {
  const int16Samples = new Int16Array(float32Data.length)
  for (let i = 0; i < float32Data.length; i++) {
    const s = float32Data[i]
    const clamped = s > 1 ? 1 : s < -1 ? -1 : s
    int16Samples[i] = clamped < 0 ? clamped * 0x8000 : clamped * 0x7FFF
  }
  // Copy int16 buffer into Uint8Array for Discord transmission
  return new Uint8Array(int16Samples.buffer)
}

/**
 * Process Discord user audio through the complete pipeline
 * Input: 48kHz PCM → Transcribe → Generate → Synthesize → Resample → Output: 48kHz PCM
 *
 * @param {Buffer|Uint8Array} pcmBuffer - Raw 48kHz mono 16-bit PCM audio
 * @param {number} sampleRate - Input sample rate (should be 48000 for Discord)
 * @param {string} userId - Discord user ID for context
 * @returns {Promise<Uint8Array>} 48kHz mono 16-bit PCM ready for Discord output
 * @throws {Error} With context (step name, userId, input size)
 */
export async function processUserAudio(pcmBuffer, sampleRate, userId) {
  if (!pcmBuffer || pcmBuffer.length === 0) {
    throw new Error(`step=input userId=${userId}: empty audio buffer`)
  }

  if (typeof sampleRate !== 'number' || sampleRate < 8000 || sampleRate > 48000) {
    throw new Error(`step=validate userId=${userId}: sampleRate ${sampleRate} out of range`)
  }

  if (!globalVoiceEmbedding) {
    throw new Error(`step=voiceEmbed userId=${userId}: no voice embedding loaded`)
  }

  try {
    // Step 1: Transcribe 48kHz PCM to text
    const transcription = await transcribe(pcmBuffer, sampleRate)
    if (transcription.confidence < 0.01) {
      console.log(`[processor] userId=${userId} transcription confidence very low: ${transcription.confidence}`)
    }

    // Step 2: Generate response from transcribed text
    const responseText = generateResponse(transcription.text, userId)

    // Step 3: Ensure TTS models are loaded, then synthesize
    await ensureTtsModels()
    const ttsOutput = await ttsOnnx.synthesize(responseText, globalVoiceEmbedding, TTS_MODELS_DIR)
    if (!ttsOutput || ttsOutput.length === 0) {
      throw new Error(`step=synthesize userId=${userId}: empty synthesis output`)
    }

    // Step 4: Resample 24kHz → 48kHz
    const resampled = resampleAudio(ttsOutput, SAMPLE_RATE_TTS, SAMPLE_RATE_DISCORD)
    if (!resampled || resampled.length === 0) {
      throw new Error(`step=resample userId=${userId}: empty resampled output`)
    }

    // Step 5: Convert float32 to int16 PCM Uint8Array
    const pcmOutput = float32ToInt16PCM(resampled)

    console.log(`[processor] userId=${userId} pipeline complete: "${responseText.slice(0, 50)}..." -> ${pcmOutput.length} bytes`)
    return pcmOutput
  } catch (err) {
    if (err.message.includes('step=')) {
      // Already contextualized error from above
      throw err
    }
    // Re-throw with step context
    throw new Error(`step=pipeline userId=${userId}: ${err.message}`)
  }
}

export default { processUserAudio, setVoiceEmbedding }
