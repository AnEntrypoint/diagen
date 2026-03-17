const worker = new Worker('./worker.js', { type: 'module' })
const ttsWorker = new Worker('./tts-worker.js', { type: 'module' })
const SpeechRecognition = window.SpeechRecognition ?? window.webkitSpeechRecognition
const synth = window.speechSynthesis

const $ = (id) => document.getElementById(id)
const history = []
let pendingResolvers = {}
let msgId = 0
let appState = 'loading'
let modelReady = false
let ttsReady = false
let ttsLoading = true
let ttsReadyResolvers = []
let ttsChunks = null
let ttsChunkResolve = null
let ttsChunkReject = null
let recognition = null

function nextId() { return ++msgId }

function setState(s) {
  appState = s
  $('status').textContent = { loading: 'Loading model…', idle: 'Ready — click Speak', recording: 'Listening…', generating: 'Thinking…', speaking: 'Speaking…', mic_only: 'Model loading — mic ready' }[s] ?? s
  $('speak-btn').disabled = s === 'generating' || s === 'speaking' || s === 'no_mic'
  $('speak-btn').classList.toggle('recording', s === 'recording')
}

function addBubble(role, text) {
  const div = document.createElement('div')
  div.className = `bubble ${role}`
  div.textContent = text
  const chat = $('chat')
  chat.appendChild(div)
  chat.scrollTop = chat.scrollHeight
  return div
}

function sendWorker(msg) {
  return new Promise((resolve, reject) => {
    const id = nextId()
    pendingResolvers[id] = { resolve, reject }
    worker.postMessage({ ...msg, id })
  })
}

worker.onmessage = (e) => {
  const { type, id, progress, message, token } = e.data
  if (type === 'progress') {
    const pct = progress?.progress ?? 0
    $('progress-bar').style.width = `${Math.round(pct)}%`
    $('progress-text').textContent = progress?.file ? `Loading ${progress.file}…` : 'Loading model…'
    return
  }
  if (type === 'token') {
    const last = $('chat').querySelector('.bubble.assistant:last-child')
    if (last) last.textContent += token
    return
  }
  const r = pendingResolvers[id]
  if (!r) return
  delete pendingResolvers[id]
  if (type === 'error') r.reject(new Error(message))
  else r.resolve(e.data)
}

function waitForTTS(timeoutMs = 10000) {
  if (ttsReady) return Promise.resolve(true)
  if (!ttsLoading) return Promise.resolve(false)
  return new Promise((resolve) => {
    const timer = setTimeout(() => {
      const idx = ttsReadyResolvers.indexOf(resolve)
      if (idx !== -1) ttsReadyResolvers.splice(idx, 1)
      resolve(false)
    }, timeoutMs)
    ttsReadyResolvers.push((ok) => { clearTimeout(timer); resolve(ok) })
  })
}

ttsWorker.onmessage = (e) => {
  const { type } = e.data
  if (type === 'loaded') {
    ttsReady = true
    ttsLoading = false
    console.log('[TTS] PocketTTS ready')
    ttsReadyResolvers.forEach(r => r(true))
    ttsReadyResolvers = []
  } else if (type === 'error' && !ttsReady && ttsLoading && !ttsChunkReject) {
    ttsLoading = false
    console.warn('[TTS] Worker failed to load, will use browser TTS:', e.data.error)
    ttsReadyResolvers.forEach(r => r(false))
    ttsReadyResolvers = []
  } else if (type === 'audio_chunk') {
    if (ttsChunks) ttsChunks.push(new Float32Array(e.data.data))
  } else if (type === 'stream_ended') {
    if (ttsChunkResolve) {
      ttsChunkResolve(ttsChunks)
      ttsChunks = null; ttsChunkResolve = null; ttsChunkReject = null
    }
  } else if (type === 'error') {
    if (ttsChunkReject) {
      ttsChunkReject(new Error(e.data.error))
      ttsChunks = null; ttsChunkResolve = null; ttsChunkReject = null
    } else {
      console.error('[TTS] Worker error (not during generation):', e.data.error)
    }
  }
}

ttsWorker.postMessage({ type: 'load' })

async function loadModel() {
  if (!SpeechRecognition) {
    $('status').textContent = 'Web Speech API not supported in this browser (use Chrome/Edge)'
    $('speak-btn').disabled = true
    return
  }
  setState('mic_only')
  $('progress-wrap').hidden = false
  try {
    await sendWorker({ type: 'load' })
    modelReady = true
    $('progress-wrap').hidden = true
    setState('idle')
  } catch (err) {
    $('progress-wrap').hidden = true
    $('status').textContent = `Model load failed: ${err.message} — mic still available`
  }
}

function playAudio(pcm, sampleRate) {
  const ctx = new AudioContext({ sampleRate })
  const buf = ctx.createBuffer(1, pcm.length, sampleRate)
  buf.copyToChannel(pcm, 0)
  const src = ctx.createBufferSource()
  src.buffer = buf
  src.connect(ctx.destination)
  return new Promise((resolve) => {
    src.onended = () => { ctx.close(); resolve() }
    ctx.resume().then(() => src.start())
  })
}

async function speak(text) {
  const ready = await waitForTTS()
  if (ready) {
    try {
      const chunks = await new Promise((resolve, reject) => {
        ttsChunks = []
        ttsChunkResolve = resolve
        ttsChunkReject = reject
        ttsWorker.postMessage({ type: 'generate', data: { text } })
      })
      if (chunks.length > 0) {
        const total = chunks.reduce((s, c) => s + c.length, 0)
        const pcm = new Float32Array(total)
        let off = 0
        for (const c of chunks) { pcm.set(c, off); off += c.length }
        await playAudio(pcm, 24000)
        return
      }
      console.warn('[TTS] PocketTTS returned 0 chunks, falling back to browser TTS')
    } catch (err) {
      console.warn('[TTS] PocketTTS failed, falling back to browser TTS:', err.message)
    }
  }
  if (!synth) return
  return new Promise((resolve) => {
    synth.cancel()
    const utt = new SpeechSynthesisUtterance(text)
    utt.onend = resolve
    utt.onerror = resolve
    synth.speak(utt)
  })
}

function startRecognition() {
  return new Promise((resolve, reject) => {
    const rec = new SpeechRecognition()
    recognition = rec
    rec.lang = 'en-US'
    rec.interimResults = true
    rec.continuous = false
    let final = ''
    rec.onresult = (e) => {
      final = Array.from(e.results).map(r => r[0].transcript).join('')
      $('status').textContent = `Listening… ${final}`
    }
    rec.onend = () => resolve(final.trim())
    rec.onerror = (e) => reject(new Error(e.error))
    rec.start()
  })
}

$('speak-btn').addEventListener('click', async () => {
  if (appState === 'generating' || appState === 'speaking' || appState === 'recording') return
  setState('recording')
  let transcript = ''
  try {
    transcript = await startRecognition()
  } catch (err) {
    $('status').textContent = `Mic error: ${err.message}`
    setState(modelReady ? 'idle' : 'mic_only')
    return
  }
  if (!transcript) { setState(modelReady ? 'idle' : 'mic_only'); return }
  addBubble('user', transcript)
  if (!modelReady) {
    $('status').textContent = 'Model still loading — please wait and try again'
    return
  }
  history.push({ role: 'user', content: transcript })
  setState('generating')
  addBubble('assistant', '')
  try {
    const { text } = await sendWorker({ type: 'generate', messages: [...history], config: { maxNewTokens: 300, temperature: 0.7 } })
    const cleaned = text.trim()
    history.push({ role: 'assistant', content: cleaned })
    const last = $('chat').querySelector('.bubble.assistant:last-child')
    if (last) last.textContent = cleaned
    setState('speaking')
    await speak(cleaned)
  } catch (err) {
    $('status').textContent = `Error: ${err.message}`
  }
  setState('idle')
})

loadModel()
