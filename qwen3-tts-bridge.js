import { spawn } from 'child_process'
import path from 'path'
import { fileURLToPath } from 'url'
import readline from 'readline'

const __dirname = path.dirname(fileURLToPath(import.meta.url))
const SERVER_SCRIPT = path.join(__dirname, 'qwen3_tts_server.py')
const STARTUP_TIMEOUT_MS = Number(process.env.QWEN3_TTS_STARTUP_MS || 1200000)
const SYNTH_TIMEOUT_MS = Number(process.env.QWEN3_TTS_SYNTH_MS || 120000)

let ttsProcess = null
let processReady = false
let startPromise = null
let rl = null
const pending = new Map()
let nextId = 1
let queueChain = Promise.resolve()

function dispatch(obj) {
  const id = obj.id
  const p = pending.get(id)
  if (!p) return
  if (obj.chunk) { p.onChunk?.(obj); return }
  pending.delete(id)
  if (obj.done) { p.resolveDone?.(obj); return }
  if (obj.success) p.resolve(obj)
  else p.reject(new Error(`qwen3-tts: ${obj.error || 'unknown error'}`))
}

function startTtsProcess() {
  if (processReady) return Promise.resolve()
  if (startPromise) return startPromise
  startPromise = new Promise((resolve, reject) => {
    const pythonCmd = process.env.QWEN3_TTS_PYTHON || 'python'
    ttsProcess = spawn(pythonCmd, ['-u', SERVER_SCRIPT], {
      stdio: ['pipe', 'pipe', 'pipe'],
      env: { ...process.env, PYTHONUNBUFFERED: '1' },
    })
    let stderrBuf = ''
    const readyTimeout = setTimeout(() => {
      if (ttsProcess) ttsProcess.kill()
      reject(new Error(`qwen3-tts startup timeout (${STARTUP_TIMEOUT_MS}ms)`))
    }, STARTUP_TIMEOUT_MS)
    ttsProcess.stderr.on('data', (chunk) => {
      const s = chunk.toString()
      stderrBuf += s
      process.stderr.write('[qwen3-tts] ' + s)
      if (!processReady && stderrBuf.includes('Model ready')) {
        processReady = true
        clearTimeout(readyTimeout)
        resolve()
      }
    })
    rl = readline.createInterface({ input: ttsProcess.stdout })
    rl.on('line', (line) => {
      if (!line.trim()) return
      try { dispatch(JSON.parse(line)) } catch { console.error('[qwen3-tts] bad json:', line.slice(0, 200)) }
    })
    ttsProcess.on('error', (err) => { clearTimeout(readyTimeout); processReady = false; reject(new Error(`qwen3-tts spawn failed: ${err.message}`)) })
    ttsProcess.on('exit', (code) => {
      processReady = false; ttsProcess = null; startPromise = null
      const err = new Error(`qwen3-tts exited code=${code}`)
      for (const p of pending.values()) p.reject(err)
      pending.clear()
    })
  })
  return startPromise
}

function enqueue(fn) {
  const prev = queueChain
  queueChain = prev.then(fn, fn)
  return queueChain
}

function decodeAudio(obj) {
  const buf = Buffer.from(obj.audio_b64, 'base64')
  const int16 = new Int16Array(buf.buffer, buf.byteOffset, buf.length / 2)
  const audio = new Float32Array(int16.length)
  for (let i = 0; i < int16.length; i++) audio[i] = int16[i] / 32768
  return { audio, sampleRate: obj.sample_rate || 24000 }
}

export async function synthesize(text, refAudioPath, refText, signal) {
  if (!text) throw new Error('text required')
  await startTtsProcess()
  return enqueue(() => {
    if (signal?.aborted) { console.log('[qwen3-tts] skip synth (already aborted)'); return null }
    const id = nextId++
    return new Promise((resolve, reject) => {
      const timeout = setTimeout(() => { pending.delete(id); reject(new Error(`qwen3-tts synth timeout (${SYNTH_TIMEOUT_MS}ms)`)) }, SYNTH_TIMEOUT_MS)
      const onAbort = () => { pending.delete(id); clearTimeout(timeout); resolve(null) }
      signal?.addEventListener?.('abort', onAbort, { once: true })
      pending.set(id, {
        resolve: (obj) => { clearTimeout(timeout); signal?.removeEventListener?.('abort', onAbort); resolve(decodeAudio(obj)) },
        reject: (err) => { clearTimeout(timeout); signal?.removeEventListener?.('abort', onAbort); reject(err) },
      })
      ttsProcess.stdin.write(JSON.stringify({ id, text, ref_audio_path: refAudioPath || null, ref_text: refText || '', streaming: false }) + '\n')
    })
  })
}

export async function synthesizeStream(text, refAudioPath, refText, onChunk, signal) {
  if (!text) throw new Error('text required')
  if (typeof onChunk !== 'function') throw new Error('onChunk required for streaming')
  await startTtsProcess()
  return enqueue(() => {
    if (signal?.aborted) { console.log('[qwen3-tts] skip stream (already aborted)'); return { sampleRate: 24000, aborted: true } }
    const id = nextId++
    return new Promise((resolve, reject) => {
      const timeout = setTimeout(() => { pending.delete(id); reject(new Error(`qwen3-tts stream timeout (${SYNTH_TIMEOUT_MS}ms)`)) }, SYNTH_TIMEOUT_MS)
      const onAbort = () => { pending.delete(id); clearTimeout(timeout); resolve({ sampleRate: 24000, aborted: true }) }
      signal?.addEventListener?.('abort', onAbort, { once: true })
      pending.set(id, {
        onChunk: (obj) => { if (signal?.aborted) return; const { audio, sampleRate } = decodeAudio(obj); onChunk(audio, sampleRate) },
        resolveDone: (obj) => { clearTimeout(timeout); signal?.removeEventListener?.('abort', onAbort); resolve({ sampleRate: obj.sample_rate || 24000 }) },
        reject: (err) => { clearTimeout(timeout); signal?.removeEventListener?.('abort', onAbort); reject(err) },
        resolve: () => {},
      })
      ttsProcess.stdin.write(JSON.stringify({ id, text, ref_audio_path: refAudioPath || null, ref_text: refText || '', streaming: true }) + '\n')
    })
  })
}

export function shutdown() {
  if (ttsProcess) { ttsProcess.kill(); ttsProcess = null; processReady = false; startPromise = null }
}

export default { synthesize, synthesizeStream, shutdown }
