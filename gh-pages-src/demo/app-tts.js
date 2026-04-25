import { setMouthOpen } from './vrm-viewer.js'

const ttsWorker = new Worker('./tts-worker.js?v=7', { type: 'module' })
let ttsReady = false, ttsLoading = false
let ttsReadyResolvers = [], ttsStreamResolve = null, ttsStreamReject = null
let audioCtx = null, nextAt = 0, analyser = null, rafId = null

const $ = (id) => document.getElementById(id)

function ensureAudioCtx() {
  if (!audioCtx || audioCtx.state === 'closed') {
    audioCtx = new AudioContext({ sampleRate: 24000 })
    analyser = audioCtx.createAnalyser(); analyser.fftSize = 256
    analyser.connect(audioCtx.destination)
  }
  return audioCtx.resume()
}

function scheduleChunk(pcm) {
  const buf = audioCtx.createBuffer(1, pcm.length, 24000)
  buf.copyToChannel(pcm, 0)
  const src = audioCtx.createBufferSource()
  src.buffer = buf
  src.connect(analyser)
  const at = Math.max(nextAt, audioCtx.currentTime)
  src.start(at)
  nextAt = at + pcm.length / 24000
  return src
}

function startMouthTick() {
  if (rafId) return
  const data = new Uint8Array(analyser.frequencyBinCount)
  const tick = () => {
    analyser.getByteTimeDomainData(data)
    let sum = 0; for (let i = 0; i < data.length; i++) { const s = (data[i] - 128) / 128; sum += s * s }
    setMouthOpen(Math.min(1, Math.sqrt(sum / data.length) * 6))
    rafId = requestAnimationFrame(tick)
  }
  rafId = requestAnimationFrame(tick)
}

function stopMouthTick() {
  if (rafId) { cancelAnimationFrame(rafId); rafId = null }
  setMouthOpen(0)
}

function ttsLoadFailed(msg) {
  ttsLoading = false; $('progress-text').textContent = msg
  setTimeout(() => { $('progress-wrap').hidden = true }, 5000)
  ttsReadyResolvers.forEach(r => r(false)); ttsReadyResolvers = []
}

ttsWorker.onmessage = async (e) => {
  const { type } = e.data
  if (type === 'voices_loaded') {
    const sel = $('voice-select'); sel.innerHTML = ''
    for (const v of e.data.voices) {
      const opt = Object.assign(document.createElement('option'), { value: v, textContent: v.charAt(0).toUpperCase() + v.slice(1) })
      if (v === e.data.defaultVoice) opt.selected = true
      sel.appendChild(opt)
    }
    $('voice-wrap').hidden = e.data.voices.length <= 1
    ttsWorker.postMessage({ type: 'load_voice', voice: e.data.defaultVoice })
  } else if (type === 'loaded') {
    ttsReady = true; ttsLoading = false
    $('progress-wrap').hidden = true
    ttsReadyResolvers.forEach(r => r(true)); ttsReadyResolvers = []
  } else if (type === 'status' && ttsLoading && !ttsReady) {
    $('progress-wrap').hidden = false; $('progress-text').textContent = e.data.status
  } else if (type === 'error' && !ttsReady && ttsLoading && !ttsStreamReject) {
    ttsLoadFailed('TTS load failed: ' + e.data.error)
  } else if (type === 'audio_chunk') {
    await ensureAudioCtx()
    startMouthTick()
    scheduleChunk(new Float32Array(e.data.data))
  } else if (type === 'stream_ended' && ttsStreamResolve) {
    const resolve = ttsStreamResolve
    ttsStreamResolve = null; ttsStreamReject = null
    const remaining = nextAt - audioCtx.currentTime
    if (remaining > 0) {
      setTimeout(() => { stopMouthTick(); resolve() }, remaining * 1000 + 100)
    } else {
      stopMouthTick(); resolve()
    }
  } else if (type === 'error') {
    if (ttsStreamReject) { stopMouthTick(); ttsStreamReject(new Error(e.data.error)); ttsStreamResolve = null; ttsStreamReject = null }
    else console.error('[TTS] error:', e.data.error)
  }
}
ttsWorker.onerror = (e) => {
  if (!ttsReady && ttsLoading) ttsLoadFailed('TTS worker crashed: ' + e.message)
}

export function startTTS() { ttsLoading = true; ttsWorker.postMessage({ type: 'load' }) }
export function isTtsReady() { return ttsReady }
export function onVoiceChange(voice) { ttsWorker.postMessage({ type: 'load_voice', voice }) }
export function waitForTTS() {
  if (ttsReady) return Promise.resolve(true)
  if (!ttsLoading) return Promise.resolve(false)
  return new Promise(resolve => { ttsReadyResolvers.push(resolve) })
}
export function stopSpeak() {
  stopMouthTick()
  if (audioCtx) { audioCtx.suspend(); nextAt = 0 }
  if (ttsStreamReject) { const rej = ttsStreamReject; ttsStreamResolve = null; ttsStreamReject = null; rej(new Error('interrupted')) }
  ttsWorker.postMessage({ type: 'cancel' })
}
export async function speak(text) {
  if (ttsLoading && !ttsReady) $('status').textContent = 'TTS loading… (first run takes ~1 min)'
  const ready = await waitForTTS()
  if (!ready) { $('status').textContent = 'TTS unavailable'; return }
  nextAt = 0
  await new Promise((resolve, reject) => {
    ttsStreamResolve = resolve; ttsStreamReject = reject
    ttsWorker.postMessage({ type: 'generate', data: { text, voice: $('voice-select').value } })
  })
}
