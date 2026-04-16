import fs from 'fs'
import path from 'path'
import { fileURLToPath } from 'url'
import { transcribe } from './discord-whisper.js'
import { generate as generateLLM, isAvailable } from './llm-ollama.js'

const __dirname = path.dirname(fileURLToPath(import.meta.url))
const VOICE_REF = path.join(__dirname, 'voices', 'cleetus.wav')

async function testSTTandLLM() {
  console.log('[test] Testing STT and LLM components separately...\n')

  // Test 1: Check if Ollama is available
  console.log('[test] 1. Checking Ollama availability...')
  const available = await isAvailable()
  console.log(`[test]    Ollama: ${available ? '✓ Ready' : '✗ Not ready'}`)
  if (!available) {
    console.log('[test]    Waiting 3 seconds for Ollama to start...')
    await new Promise(r => setTimeout(r, 3000))
  }

  // Test 2: Test Whisper with cleetus.wav
  console.log('\n[test] 2. Testing Whisper STT with cleetus.wav...')
  if (!fs.existsSync(VOICE_REF)) {
    console.error(`[test]    ✗ Voice file not found: ${VOICE_REF}`)
    return false
  }

  try {
    const wavData = fs.readFileSync(VOICE_REF)
    console.log(`[test]    Loaded: ${wavData.length} bytes`)
    console.log(`[test]    Transcribing...`)
    const result = await transcribe(wavData, 24000) // Try 24kHz (common for TTS)
    console.log(`[test]    ✓ Whisper result: "${result.text}"`)
    console.log(`[test]    Confidence: ${(result.confidence * 100).toFixed(1)}%`)
  } catch (err) {
    console.error(`[test]    ✗ Whisper failed: ${err.message}`)
    return false
  }

  // Test 3: Test LLM
  console.log('\n[test] 3. Testing Llama LLM...')
  try {
    const response = await generateLLM('Hello, who are you?')
    console.log(`[test]    ✓ LLM response: "${response.slice(0, 100)}${response.length > 100 ? '...' : ''}"`)
  } catch (err) {
    console.error(`[test]    ✗ LLM failed: ${err.message}`)
    return false
  }

  console.log('\n[test] ✓ All tests passed!')
  return true
}

testSTTandLLM()
  .then(success => process.exit(success ? 0 : 1))
  .catch(err => {
    console.error('[test] Fatal:', err.message)
    process.exit(1)
  })
