import { spawn } from 'child_process'
import fs from 'fs'
import path from 'path'
import { fileURLToPath } from 'url'

const __dirname = path.dirname(fileURLToPath(import.meta.url))
const OMNIVOICE_REPO = 'C:\\dev\\omnivoice'
const SERVER_SCRIPT = path.join(__dirname, 'omnivoice_tts_server.py')

let ttsProcess = null
let processReady = false
let readyTimeout = null

function startTtsProcess() {
  if (ttsProcess) return Promise.resolve()

  return new Promise((resolve, reject) => {
    const HF_CACHE = path.join(__dirname, 'models', 'omnivoice')
    fs.mkdirSync(HF_CACHE, { recursive: true })
    ttsProcess = spawn('uv', ['run', 'python', SERVER_SCRIPT], {
      cwd: OMNIVOICE_REPO,
      stdio: ['pipe', 'pipe', 'pipe'],
      env: { ...process.env, HF_HOME: HF_CACHE, HUGGINGFACE_HUB_CACHE: HF_CACHE, TRANSFORMERS_CACHE: HF_CACHE },
    })

    let stderrOutput = ''
    readyTimeout = setTimeout(() => {
      readyTimeout = null
      if (ttsProcess) ttsProcess.kill()
      reject(new Error('OmniVoice startup timeout (600s)'))
    }, 600000)

    ttsProcess.stderr.on('data', (chunk) => {
      stderrOutput += chunk.toString()
      process.stderr.write('[omnivoice] ' + chunk.toString())
      if (stderrOutput.includes('[omnivoice] Model ready')) {
        if (readyTimeout) { clearTimeout(readyTimeout); readyTimeout = null }
        processReady = true
        resolve()
      }
    })

    ttsProcess.on('error', (err) => {
      processReady = false
      if (readyTimeout) { clearTimeout(readyTimeout); readyTimeout = null }
      reject(new Error(`TTS process spawn failed: ${err.message}`))
    })

    ttsProcess.on('exit', (code) => {
      processReady = false
      ttsProcess = null
      if (readyTimeout) { clearTimeout(readyTimeout); readyTimeout = null }
    })
  })
}

export async function synthesize(text, refAudioPath, refText) {
  if (!text) throw new Error('text required for synthesis')
  await startTtsProcess()

  return new Promise((resolve, reject) => {
    let responseData = ''
    const timeout = setTimeout(() => {
      ttsProcess.stdout.removeListener('data', onData)
      reject(new Error('TTS synthesis timeout (300s)'))
    }, 300000)

    function onData(chunk) {
      responseData += chunk.toString()
      if (!responseData.includes('\n')) return
      ttsProcess.stdout.removeListener('data', onData)
      clearTimeout(timeout)
      try {
        const lines = responseData.trim().split('\n')
        const response = JSON.parse(lines[lines.length - 1])
        if (!response.success) throw new Error(`TTS failed: ${response.error}`)
        const audioBuffer = Buffer.from(response.audio_b64, 'base64')
        const int16 = new Int16Array(audioBuffer.buffer, audioBuffer.byteOffset, audioBuffer.length / 2)
        const audio = new Float32Array(int16.length)
        for (let i = 0; i < int16.length; i++) audio[i] = int16[i] / 32768
        resolve({ audio, sampleRate: response.sample_rate || 24000 })
      } catch (err) {
        reject(new Error(`TTS parse failed: ${err.message}`))
      }
    }
    ttsProcess.stdout.on('data', onData)

    try {
      const refAudioB64 = refAudioPath ? fs.readFileSync(refAudioPath).toString('base64') : null
      ttsProcess.stdin.write(JSON.stringify({ text, ref_audio_b64: refAudioB64, ref_text: refText }) + '\n')
    } catch (err) {
      clearTimeout(timeout)
      reject(new Error(`TTS request send failed: ${err.message}`))
    }
  })
}

export function shutdown() {
  if (ttsProcess) { ttsProcess.kill(); ttsProcess = null; processReady = false }
}

export default { synthesize, shutdown }
