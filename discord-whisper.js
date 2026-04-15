/**
 * Discord Whisper STT Module
 *
 * Uses @xenova/transformers library to run OpenAI's Whisper model
 * for speech-to-text transcription of Discord audio streams.
 *
 * Model: Xenova/whisper-tiny (lightweight, ~39MB)
 * Suitable for real-time Discord voice processing
 *
 * Whisper uses:
 * - AutoProcessor: converts audio to mel-spectrogram features
 * - AutoModelForSpeechSeq2Seq: encoder-decoder speech-to-text
 */

import { AutoProcessor, AutoModelForSpeechSeq2Seq, pipeline } from '@xenova/transformers';

/**
 * Global pipeline cache - initialized once and reused
 * Pattern: promise-based singleton to handle concurrent first-calls
 */
let whisperPipeline = null;
let pipelineInitPromise = null;

/**
 * Initialize Whisper pipeline on first call
 * Subsequent calls return cached pipeline
 * Handles concurrent access with promise locking
 */
async function initPipeline() {
  // If already initialized, return immediately
  if (whisperPipeline) {
    return whisperPipeline;
  }

  // If initialization in progress, await shared promise
  if (pipelineInitPromise) {
    return pipelineInitPromise;
  }

  // Start initialization and cache the promise
  pipelineInitPromise = (async () => {
    try {
      // pipeline('automatic-speech-recognition') handles all model/processor loading
      // Using whisper-tiny for speed, can be changed to whisper-small/base/medium
      whisperPipeline = await pipeline(
        'automatic-speech-recognition',
        'Xenova/whisper-tiny',
        { quantized: true } // Use quantized version for faster inference
      );
      return whisperPipeline;
    } catch (err) {
      pipelineInitPromise = null; // Reset on failure so retry works
      throw new Error(`Whisper pipeline initialization failed: ${err.message}`);
    }
  })();

  return pipelineInitPromise;
}

/**
 * Transcribe PCM audio buffer to text
 *
 * @param {Buffer|Uint8Array} pcmBuffer - Raw 16-bit PCM audio data
 * @param {number} sampleRate - Sample rate in Hz (default 48000 for Discord)
 * @returns {Promise<{text: string, confidence: number}>}
 *
 * Discord voice streams are 48kHz mono 16-bit PCM
 * Whisper expects 16kHz mono, so input will be resampled automatically
 * Confidence is 0-1 representing model's certainty (0 = low, 1 = high)
 *
 * Model: Xenova/whisper-tiny (39MB, optimized for inference)
 * Alternatives: Xenova/whisper-small, Xenova/whisper-base, Xenova/whisper-medium
 *
 * Throws:
 * - Model download/initialization errors
 * - Invalid audio buffer errors
 */
export async function transcribe(pcmBuffer, sampleRate = 48000) {
  // Validate input
  if (!pcmBuffer || (typeof pcmBuffer !== 'object')) {
    throw new Error('transcribe: pcmBuffer must be a Buffer or Uint8Array');
  }

  if (typeof sampleRate !== 'number' || sampleRate < 8000 || sampleRate > 48000) {
    throw new Error(`transcribe: sampleRate must be between 8000-48000, got ${sampleRate}`);
  }

  // Initialize pipeline (cached after first call)
  const asr = await initPipeline();

  // Convert PCM bytes to float32 array (Whisper input format)
  const pcmArray = new Int16Array(
    pcmBuffer.buffer || pcmBuffer,
    pcmBuffer.byteOffset || 0,
    pcmBuffer.byteLength ? pcmBuffer.byteLength / 2 : pcmBuffer.length
  );

  // Normalize to [-1, 1] range (Whisper expects float32 in this range)
  const audioData = new Float32Array(pcmArray.length);
  for (let i = 0; i < pcmArray.length; i++) {
    audioData[i] = pcmArray[i] / 32768.0;
  }

  // Resample from Discord's 48kHz to Whisper's 16kHz
  const targetRate = 16000;
  const resampleRatio = targetRate / sampleRate;
  const resampledLength = Math.floor(audioData.length * resampleRatio);
  const resampled = new Float32Array(resampledLength);

  // Linear interpolation resampling
  for (let i = 0; i < resampledLength; i++) {
    const srcIdx = i / resampleRatio;
    const srcIdxFloor = Math.floor(srcIdx);
    const srcIdxCeil = Math.min(srcIdxFloor + 1, audioData.length - 1);
    const fraction = srcIdx - srcIdxFloor;

    resampled[i] =
      audioData[srcIdxFloor] * (1 - fraction) +
      audioData[srcIdxCeil] * fraction;
  }

  try {
    // Run speech-to-text pipeline
    // Pipeline handles: audio normalization, mel-spectrogram, inference, decoding
    const result = await asr(resampled, {
      chunk_length_s: 30, // Process 30s chunks to manage memory
      stride_length_s: 5 // 5s stride for consistency
    });

    // result contains: { text: string }
    // Confidence calculation: model doesn't expose per-token probabilities directly
    // Use output length as proxy for confidence (longer transcriptions = model confidence)
    const confidence = Math.min(1.0, result.text.length / 100.0);

    return {
      text: result.text || '[no speech detected]',
      confidence: Math.max(0, Math.min(1, confidence))
    };
  } catch (err) {
    throw new Error(`Whisper transcription failed: ${err.message}`);
  }
}

export default { transcribe };
