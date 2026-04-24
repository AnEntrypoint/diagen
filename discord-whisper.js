import { pipeline, env } from '@huggingface/transformers'
import path from 'path'
import { fileURLToPath } from 'url'

const __dirname = path.dirname(fileURLToPath(import.meta.url))
env.cacheDir = path.join(__dirname, 'models', 'whisper')
env.localModelPath = path.join(__dirname, 'models', 'whisper')
env.allowRemoteModels = true

let whisperPipeline = null
let pipelineInitPromise = null

async function initPipeline() {
  if (whisperPipeline) return whisperPipeline
  if (pipelineInitPromise) return pipelineInitPromise

  pipelineInitPromise = (async () => {
    try {
      whisperPipeline = await pipeline('automatic-speech-recognition', 'Xenova/whisper-base', { dtype: 'q8' })
      return whisperPipeline
    } catch (err) {
      pipelineInitPromise = null
      throw new Error(`Whisper pipeline initialization failed: ${err.message}`)
    }
  })()

  return pipelineInitPromise
}

export async function transcribe(pcmBuffer, sampleRate = 48000) {
  if (!pcmBuffer || typeof pcmBuffer !== 'object') throw new Error('transcribe: pcmBuffer must be a Buffer or Uint8Array')
  if (typeof sampleRate !== 'number' || sampleRate < 8000 || sampleRate > 48000) throw new Error(`transcribe: sampleRate must be between 8000-48000, got ${sampleRate}`)

  const asr = await initPipeline()

  const pcmArray = new Int16Array(
    pcmBuffer.buffer || pcmBuffer,
    pcmBuffer.byteOffset || 0,
    pcmBuffer.byteLength ? pcmBuffer.byteLength / 2 : pcmBuffer.length
  )

  const audioData = new Float32Array(pcmArray.length)
  for (let i = 0; i < pcmArray.length; i++) audioData[i] = pcmArray[i] / 32768.0

  const targetRate = 16000
  const resampleRatio = targetRate / sampleRate
  const resampledLength = Math.floor(audioData.length * resampleRatio)
  const resampled = new Float32Array(resampledLength)

  for (let i = 0; i < resampledLength; i++) {
    const srcIdx = i / resampleRatio
    const srcIdxFloor = Math.floor(srcIdx)
    const srcIdxCeil = Math.min(srcIdxFloor + 1, audioData.length - 1)
    const fraction = srcIdx - srcIdxFloor
    resampled[i] = audioData[srcIdxFloor] * (1 - fraction) + audioData[srcIdxCeil] * fraction
  }

  try {
    const result = await asr(resampled, {
      chunk_length_s: 30,
      stride_length_s: 5,
      language: 'english',
      task: 'transcribe',
      no_speech_threshold: 0.2,
      condition_on_previous_text: false,
      initial_prompt: 'Cleetus, the gas station owner, talking with customers.',
    })
    const confidence = Math.min(1.0, result.text.length / 100.0)
    return { text: result.text || '[no speech detected]', confidence: Math.max(0, Math.min(1, confidence)) }
  } catch (err) {
    throw new Error(`Whisper transcription failed: ${err.message}`)
  }
}

export default { transcribe }
