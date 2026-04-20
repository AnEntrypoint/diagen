import { spawn } from 'child_process'
import fs from 'fs'
import path from 'path'
import { fileURLToPath } from 'url'
import readline from 'readline'

const __dirname = path.dirname(fileURLToPath(import.meta.url))
const SERVER_SCRIPT = path.join(__dirname, 'pocket_tts_server.py')
const HF_CACHE = path.join(__dirname, 'models', 'pocket-tts')

let ttsProcess = null
let processReady = false
let startPromise = null
let rl = null
const pending = new Map()
let nextId = 1

function dispatch(obj) {
  const id = obj.id
  const p = pending.get(id)
  if (!p) return
  if (obj.chunk) { p.onChunk?.(obj); return }
  pending.delete(id)
  if (obj.done) { p.resolveDone?.(obj); return }
  if (obj.success) p.resolve(obj)
  else p.reject(new Error(`pocket-tts: ${obj.error || 'unknown error'}`))
}

function startTtsProcess() {
  if (processReady) return Promise.resolve()
  if (startPromise) return startPromise
  startPromise = new Promise((resolve, reject) => {
    fs.mkdirSync(HF_CACHE, { recursive: true })
    const pythonCmd = process.env.POCKET_TTS_PYTHON || 'python'
    ttsProcess = spawn(pythonCmd, [SERVER_SCRIPT], {
      stdio: ['pipe', 'pipe', 'pipe'],
      env: { ...process.env, HF_HOME: HF_CACHE, HUGGINGFACE_HUB_CACHE: HF_CACHE, TRANSFORMERS_CACHE: HF_CACHE, PYTHONUNBUFFERED: '1' },
    })
    let stderrBuf = ''
    const readyTimeout = setTimeout(() => {
      if (ttsProcess) ttsProcess.kill()
      reject(new Error('pocket-tts startup timeout (600s)'))
    }, 600000)
    ttsProcess.stderr.on('data', (chunk) => {
      const s = chunk.toString()
      stderrBuf += s
      process.stderr.write('[pocket-tts] ' + s)
      if (!processReady && stderrBuf.includes('Model ready')) {
        processReady = true
        clearTimeout(readyTimeout)
        resolve()
      }
    })
    rl = readline.createInterface({ input: ttsProcess.stdout })
    rl.on('line', (line) => {
      if (!line.trim()) return
      try { dispatch(JSON.parse(line)) } catch (err) { console.error('[pocket-tts] bad json:', line.slice(0, 200)) }
    })
    ttsProcess.on('error', (err) => { clearTimeout(readyTimeout); processReady = false; reject(new Error(`pocket-tts spawn failed: ${err.message}`)) })
    ttsProcess.on('exit', (code) => {
      processReady = false; ttsProcess = null; startPromise = null
      const err = new Error(`pocket-tts exited code=${code}`)
      for (const p of pending.values()) p.reject(err)
      pending.clear()
    })
  })
  return startPromise
}

export async function synthesize(text, refAudioPath, _refText) {
  if (!text) throw new Error('text required')
  await startTtsProcess()
  const id = nextId++
  return new Promise((resolve, reject) => {
    const timeout = setTimeout(() => { pending.delete(id); reject(new Error('pocket-tts synth timeout (120s)')) }, 120000)
    pending.set(id, {
      resolve: (obj) => { clearTimeout(timeout); resolve(decodeAudio(obj)) },
      reject: (err) => { clearTimeout(timeout); reject(err) },
    })
    ttsProcess.stdin.write(JSON.stringify({ id, text, ref_audio_path: refAudioPath || null, streaming: false }) + '\n')
  })
}

export async function synthesizeStream(text, refAudioPath, _refText, onChunk) {
  if (!text) throw new Error('text required')
  if (typeof onChunk !== 'function') throw new Error('onChunk required for streaming')
  await startTtsProcess()
  const id = nextId++
  return new Promise((resolve, reject) => {
    const timeout = setTimeout(() => { pending.delete(id); reject(new Error('pocket-tts stream timeout (120s)')) }, 120000)
    pending.set(id, {
      onChunk: (obj) => { const { audio, sampleRate } = decodeAudio(obj); onChunk(audio, sampleRate) },
      resolveDone: (obj) => { clearTimeout(timeout); resolve({ sampleRate: obj.sample_rate || 24000 }) },
      reject: (err) => { clearTimeout(timeout); reject(err) },
      resolve: () => {},
    })
    ttsProcess.stdin.write(JSON.stringify({ id, text, ref_audio_path: refAudioPath || null, streaming: true }) + '\n')
  })
}

function decodeAudio(obj) {
  const buf = Buffer.from(obj.audio_b64, 'base64')
  const int16 = new Int16Array(buf.buffer, buf.byteOffset, buf.length / 2)
  const audio = new Float32Array(int16.length)
  for (let i = 0; i < int16.length; i++) audio[i] = int16[i] / 32768
  return { audio, sampleRate: obj.sample_rate || 24000 }
}

export function shutdown() {
  if (ttsProcess) { ttsProcess.kill(); ttsProcess = null; processReady = false; startPromise = null }
}

export default { synthesize, synthesizeStream, shutdown }
