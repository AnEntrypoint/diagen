const worker = new Worker('./worker.js', { type: 'module' })
const SpeechRecognition = window.SpeechRecognition ?? window.webkitSpeechRecognition
const synth = window.speechSynthesis

const $ = (id) => document.getElementById(id)
const history = []
let pendingResolvers = {}
let msgId = 0
let appState = 'loading'
let recognition = null

function nextId() { return ++msgId }

function setState(s) {
  appState = s
  $('status').textContent = { loading: 'Loading model…', idle: 'Ready — click Speak', recording: 'Listening…', generating: 'Thinking…', speaking: 'Speaking…' }[s] ?? s
  $('speak-btn').disabled = s !== 'idle'
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
  const { type, id, progress, message, text, token } = e.data
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
  else r.resolve({ type, text })
}

async function loadModel() {
  setState('loading')
  $('progress-wrap').hidden = false
  try {
    await sendWorker({ type: 'load' })
    $('progress-wrap').hidden = true
    if (!SpeechRecognition) {
      $('status').textContent = 'Web Speech API not supported in this browser (use Chrome/Edge)'
      return
    }
    setState('idle')
  } catch (err) {
    $('status').textContent = `Model load failed: ${err.message}`
  }
}

function speak(text) {
  return new Promise((resolve) => {
    if (!synth) { resolve(); return }
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
  if (appState !== 'idle') return
  setState('recording')
  let transcript = ''
  try {
    transcript = await startRecognition()
  } catch (err) {
    $('status').textContent = `Mic error: ${err.message}`
    setState('idle')
    return
  }
  if (!transcript) { setState('idle'); return }
  addBubble('user', transcript)
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
