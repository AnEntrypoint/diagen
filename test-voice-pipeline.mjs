import fs from 'fs'
import path from 'path'
import { fileURLToPath } from 'url'
import { synthesize } from './omnivoice-tts-bridge.js'
import { transcribe } from './discord-whisper.js'
import { generate as generateLLM } from './llm-ollama.js'
import { resampleAudio } from './server-utils.mjs'

const __dirname = path.dirname(fileURLToPath(import.meta.url))

// Test prompts
const TEST_PROMPTS = [
  'What time is it?',
  'Tell me a joke',
  'How many continents are there?',
  'Spell hello',
  'What is the capital of France?'
]

const VOICE_REF = path.join(__dirname, 'voices', 'cleetus.wav')
const RESULTS_FILE = path.join(__dirname, 'test-voice-results.json')

async function sleep(ms) {
  return new Promise(r => setTimeout(r, ms))
}

async function runTest() {
  console.log('[test] Starting voice pipeline round-trip test...')
  console.log(`[test] Testing ${TEST_PROMPTS.length} prompts`)

  const results = []

  for (let i = 0; i < TEST_PROMPTS.length; i++) {
    const prompt = TEST_PROMPTS[i]
    console.log(`\n[test] ===== PROMPT ${i + 1}/${TEST_PROMPTS.length} =====`)
    console.log(`[test] Input: "${prompt}"`)

    const testResult = {
      promptIndex: i + 1,
      originalText: prompt,
      steps: {},
      success: false,
      error: null
    }

    try {
      // Step 1: Synthesize prompt with TTS
      console.log('[test] Step 1/4: TTS synthesis...')
      const audioFloat = await synthesize(prompt, VOICE_REF, 'reference speech')
      if (!audioFloat || audioFloat.length === 0) throw new Error('TTS returned empty audio')
      console.log(`[test]   ✓ Synthesized ${audioFloat.length} samples at 24kHz`)
      testResult.steps.tts = { samples: audioFloat.length, sampleRate: 24000 }

      // Step 2: Resample to 16kHz for Whisper
      console.log('[test] Step 2/4: Resample 24kHz → 16kHz...')
      const audio16k = resampleAudio(audioFloat, 24000, 16000)
      if (!audio16k || audio16k.length === 0) throw new Error('Resampling returned empty audio')
      console.log(`[test]   ✓ Resampled to ${audio16k.length} samples at 16kHz`)
      testResult.steps.resample = { samples: audio16k.length, sampleRate: 16000 }

      // Step 3: Transcribe with Whisper
      console.log('[test] Step 3/4: Whisper STT transcription...')
      const int16 = new Int16Array(audio16k.length)
      for (let j = 0; j < audio16k.length; j++) {
        const v = Math.max(-1, Math.min(1, audio16k[j]))
        int16[j] = v < 0 ? v * 0x8000 : v * 0x7FFF
      }
      const pcmBuffer = Buffer.from(int16.buffer)
      const transcribed = await transcribe(pcmBuffer, 16000)
      console.log(`[test]   ✓ Transcribed: "${transcribed.text}"`)
      console.log(`[test]   Confidence: ${(transcribed.confidence * 100).toFixed(1)}%`)
      testResult.steps.stt = { text: transcribed.text, confidence: transcribed.confidence }

      // Compare original vs transcribed
      const match = prompt.toLowerCase() === transcribed.text.toLowerCase()
      testResult.steps.stt.matches = match
      if (!match) {
        console.log(`[test]   ⚠ Mismatch: "${prompt}" ≠ "${transcribed.text}"`)
      } else {
        console.log('[test]   ✓ Perfect match!')
      }

      // Step 4: Generate LLM response
      console.log('[test] Step 4/4: LLM response generation...')
      const response = await generateLLM(transcribed.text)
      console.log(`[test]   ✓ Generated: "${response.slice(0, 100)}${response.length > 100 ? '...' : ''}"`)
      testResult.steps.llm = { responseLength: response.length, response }

      // Bonus: Synthesize LLM response
      console.log('[test] Bonus: Synthesizing LLM response...')
      const responseAudio = await synthesize(response, VOICE_REF, 'reference speech')
      console.log(`[test]   ✓ Response audio: ${responseAudio.length} samples at 24kHz`)
      testResult.steps.responseTts = { samples: responseAudio.length, sampleRate: 24000 }

      testResult.success = true
      console.log('[test] ✓ PASS')
    } catch (err) {
      testResult.error = err.message
      console.error(`[test] ✗ FAILED: ${err.message}`)
    }

    results.push(testResult)

    // 2s delay between tests
    if (i < TEST_PROMPTS.length - 1) {
      console.log('[test] Waiting 2s before next test...')
      await sleep(2000)
    }
  }

  // Summary
  console.log(`\n[test] ===== SUMMARY =====`)
  const passed = results.filter(r => r.success).length
  console.log(`[test] Passed: ${passed}/${results.length}`)

  for (const r of results) {
    const status = r.success ? '✓' : '✗'
    const match = r.steps.stt?.matches ? '(match)' : r.steps.stt ? '(mismatch)' : ''
    console.log(`[test] ${status} #${r.promptIndex}: "${r.originalText}" ${match}`)
  }

  // Write results to file
  fs.writeFileSync(RESULTS_FILE, JSON.stringify(results, null, 2))
  console.log(`\n[test] Results saved to ${RESULTS_FILE}`)

  return passed === results.length
}

runTest()
  .then(success => {
    process.exit(success ? 0 : 1)
  })
  .catch(err => {
    console.error('[test] Fatal error:', err)
    process.exit(1)
  })
