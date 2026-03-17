const worker = new Worker('./worker.js', { type: 'module' })
const SpeechRecognition = window.SpeechRecognition ?? window.webkitSpeechRecognition
const synth = window.speechSynthesis

const $ = (id) => document.getElementById(id)
const history = []
let pendingResolvers = {}
let msgId = 0
let appState = 'loading'
let modelReady = false
let ttsReady = false
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
  const { type, id, progress, message, text, token, audio, sampling_rate } = e.data
  if (type === 'progress' || type === 'tts_progress') {
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
  else if (type === 'audio') r.resolve({ audio, sampling_rate })
  else r.resolve(e.data)
}

async function loadModel() {
  if (!SpeechRecognition) {
    $('status').textContent = 'Web Speech API not supported in this browser (use Chrome/Edge)'
    $('speak-btn').disabled = true
    return
  }
  setState('mic_only')
  $('progress-wrap').hidden = false
  try {
    const result = await sendWorker({ type: 'load' })
    modelReady = true
    ttsReady = result.ttsLoaded === true
    $('progress-wrap').hidden = true
    setState('idle')
  } catch (err) {
    $('progress-wrap').hidden = true
    $('status').textContent = `Model load failed: ${err.message} — mic still available`
  }
}

async function playAudio(audio, samplingRate) {
  const ctx = new AudioContext({ sampleRate: samplingRate })
  const buf = ctx.createBuffer(1, audio.length, samplingRate)
  buf.copyToChannel(audio, 0)
  const src = ctx.createBufferSource()
  src.buffer = buf
  src.connect(ctx.destination)
  return new Promise((resolve) => {
    src.onended = () => { ctx.close(); resolve() }
    src.start()
  })
}

async function speak(text) {
  if (ttsReady) {
    try {
      const { audio, sampling_rate } = await sendWorker({ type: 'synthesize', text })
      await playAudio(audio, sampling_rate)
      return
    } catch (err) {
      console.warn('PocketTTS failed, falling back to browser TTS:', err.message)
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
