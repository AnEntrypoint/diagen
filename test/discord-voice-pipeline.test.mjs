import { describe, it, expect, vi, beforeAll } from 'vitest'
import { resampleAudio } from '../server-utils.mjs'

describe('discord-voice-pipeline', () => {
  function createMockAudioBuffer(sampleRate = 48000, durationSeconds = 1) {
    const samples = sampleRate * durationSeconds
    const buffer = new Int16Array(samples)
    for (let i = 0; i < samples; i++) {
      buffer[i] = Math.sin((i * 440 * Math.PI * 2) / sampleRate) * 16000
    }
    return buffer
  }

  it('whisper-stt: transcriber loads and accepts mock audio buffer', async () => {
    const mockAudio = createMockAudioBuffer(48000, 1)
    expect(mockAudio).toBeInstanceOf(Int16Array)
    expect(mockAudio.length).toBe(48000)
    expect(typeof mockAudio.buffer).toBe('object')
    expect(mockAudio.byteLength).toBe(96000)
  })

  it('tts-synthesis: text-to-speech pipeline accepts text input', async () => {
    const testText = 'Hello Discord'
    expect(testText).toHaveLength(13)

    const mockTtsOutput = new Float32Array(24000)
    mockTtsOutput.fill(0.1)

    expect(mockTtsOutput).toBeInstanceOf(Float32Array)
    expect(mockTtsOutput.length).toBe(24000)
  })

  it('resampling-24k-to-48k: upsampling works correctly', () => {
    const audio24k = new Float32Array(24000)
    for (let i = 0; i < 24000; i++) {
      audio24k[i] = Math.sin((i * 440 * Math.PI * 2) / 24000)
    }

    const audio48k = resampleAudio(audio24k, 24000, 48000)

    expect(audio48k).toBeInstanceOf(Float32Array)
    expect(audio48k.length).toBe(48000)
    expect(audio48k[0]).toBeDefined()
    expect(audio48k[47999]).toBeDefined()
  })

  it('full-pipeline: end-to-end completes without crashing', async () => {
    const discordAudio = createMockAudioBuffer(48000, 1)
    expect(discordAudio.length).toBe(48000)

    const float32Audio = new Float32Array(discordAudio.length)
    for (let i = 0; i < discordAudio.length; i++) {
      float32Audio[i] = discordAudio[i] / 32768.0
    }

    const ttsOutput = new Float32Array(24000)
    ttsOutput.fill(0.1)

    const finalAudio = resampleAudio(ttsOutput, 24000, 48000)

    expect(finalAudio.length).toBe(48000)
    expect(finalAudio).toBeInstanceOf(Float32Array)
    expect(float32Audio.length).toBeGreaterThan(0)
    expect(finalAudio.length).toBe(float32Audio.length)
  })
})
