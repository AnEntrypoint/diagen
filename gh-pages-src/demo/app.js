import { initVRM, setMouthOpen, setVRMPaused } from './vrm-viewer.js'
const worker = new Worker('./worker.js?v=64', { type: 'module' })
const ttsWorker = new Worker('./tts-worker.js', { type: 'module' })
const SpeechRecognition = window.SpeechRecognition ?? window.webkitSpeechRecognition
const synth = window.speechSynthesis
const $ = (id) => document.getElementById(id)
const history = []
let pendingResolvers = {}, msgId = 0, appState = 'loading'
let modelReady = false, modelLoadDone = false, ttsReady = false, ttsLoading = false
let ttsReadyResolvers = [], ttsChunkResolve = null, ttsChunkReject = null, ttsChunks = []
let audioCtx = null, personaHistory = [], personaPrefill = null, personaDesc = '', recognition = null
function nextId() { return ++msgId }
function setState(s) {
  appState = s
  $('status').textContent = { loading: 'Loading model…', idle: 'Ready — click Speak', recording: 'Listening…', generating: 'Thinking…', speaking: 'Speaking…', mic_only: 'Model loading — mic ready' }[s] ?? s
  $('speak-btn').disabled = s === 'generating' || s === 'speaking' || s === 'no_mic'
  $('speak-btn').classList.toggle('recording', s === 'recording')
}
function addBubble(role, text) {
  const div = Object.assign(document.createElement('div'), { className: `bubble ${role}`, textContent: text })
  $('chat').appendChild(div)
  $('chat').scrollTop = $('chat').scrollHeight
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
    if (!modelReady) {
      $('progress-wrap').hidden = false
      $('progress-bar').style.width = `${Math.round(progress?.progress ?? 0)}%`
      $('progress-text').textContent = progress?.file ? `Loading ${progress.file}…` : 'Loading model…'
    }
    return
  }
  if (type === 'token') { const last = $('chat').querySelector('.bubble.assistant:last-child'); if (last) last.textContent += token; return }
  if (type === 'error' && id == null) {
    const msg = message || 'Worker error'
    Object.values(pendingResolvers).forEach(r => r.reject(new Error(msg)))
    Object.keys(pendingResolvers).forEach(k => delete pendingResolvers[k])
    return
  }
  const r = pendingResolvers[id]
  if (!r) return
  delete pendingResolvers[id]
  if (type === 'error') r.reject(new Error(message)); else r.resolve(e.data)
}
worker.onerror = (e) => {
  const msg = [e.message, e.filename, e.lineno, e.colno, String(e.error)].filter(Boolean).join(' | ') || 'Worker crashed (out of memory?)'
  Object.values(pendingResolvers).forEach(r => r.reject(new Error(msg)))
  Object.keys(pendingResolvers).forEach(k => delete pendingResolvers[k])
  $('progress-wrap').hidden = true
  $('status').textContent = `Model load failed: ${msg} — mic still available`
}
function waitForTTS() {
  if (ttsReady) return Promise.resolve(true)
  if (!ttsLoading) return Promise.resolve(false)
  return new Promise(resolve => { ttsReadyResolvers.push(resolve) })
}
function playPcm(pcm) {
  if (!audioCtx || audioCtx.state === 'closed') audioCtx = new AudioContext({ sampleRate: 24000 })
  return audioCtx.resume().then(() => {
    const buf = audioCtx.createBuffer(1, pcm.length, 24000); buf.copyToChannel(pcm, 0)
    const src = audioCtx.createBufferSource(); src.buffer = buf
    const analyser = audioCtx.createAnalyser(); analyser.fftSize = 256
    src.connect(analyser); analyser.connect(audioCtx.destination)
    const data = new Uint8Array(analyser.frequencyBinCount)
    let rafId
    const tick = () => {
      analyser.getByteTimeDomainData(data)
      let sum = 0; for (let i = 0; i < data.length; i++) { const s = (data[i] - 128) / 128; sum += s * s }
      setMouthOpen(Math.min(1, Math.sqrt(sum / data.length) * 6))
      rafId = requestAnimationFrame(tick)
    }
    tick()
    return new Promise(r => { src.onended = () => { cancelAnimationFrame(rafId); setMouthOpen(0); r() }; src.start() })
  })
}
function ttsLoadFailed(msg) {
  ttsLoading = false; $('progress-text').textContent = msg
  setTimeout(() => { $('progress-wrap').hidden = true }, 5000)
  ttsReadyResolvers.forEach(r => r(false)); ttsReadyResolvers = []
}
ttsWorker.onmessage = (e) => {
  const { type } = e.data
  if (type === 'voices_loaded') {
    const sel = $('voice-select')
    sel.innerHTML = ''
    for (const v of e.data.voices) {
      const opt = Object.assign(document.createElement('option'), { value: v, textContent: v.charAt(0).toUpperCase() + v.slice(1) })
      if (v === e.data.defaultVoice) opt.selected = true
      sel.appendChild(opt)
    }
    $('voice-wrap').hidden = e.data.voices.length <= 1
    ttsWorker.postMessage({ type: 'load_voice', voice: e.data.defaultVoice })
  } else if (type === 'loaded') {
    ttsReady = true; ttsLoading = false
    if (modelReady) $('progress-wrap').hidden = true
    ttsReadyResolvers.forEach(r => r(true)); ttsReadyResolvers = []
  } else if (type === 'status' && ttsLoading && !ttsReady) {
    $('progress-wrap').hidden = false; $('progress-text').textContent = e.data.status
  } else if (type === 'error' && !ttsReady && ttsLoading && !ttsChunkReject) {
    ttsLoadFailed('TTS load failed: ' + e.data.error)
  } else if (type === 'audio_chunk') {
    ttsChunks.push(new Float32Array(e.data.data))
  } else if (type === 'stream_ended' && ttsChunkResolve) {
    const [resolve, reject, chunks] = [ttsChunkResolve, ttsChunkReject, ttsChunks]
    ttsChunkResolve = null; ttsChunkReject = null; ttsChunks = []
    if (chunks.length === 0) { resolve(); return }
    const total = chunks.reduce((s, c) => s + c.length, 0)
    const pcm = new Float32Array(total)
    let off = 0; for (const c of chunks) { pcm.set(c, off); off += c.length }
    playPcm(pcm).then(resolve).catch(reject)
  } else if (type === 'error') {
    if (ttsChunkReject) { ttsChunkReject(new Error(e.data.error)); ttsChunkResolve = null; ttsChunkReject = null }
    else console.error('[TTS] Worker error (not during generation):', e.data.error)
  }
}
ttsWorker.onerror = (e) => {
  if (!ttsReady && ttsLoading) { $('progress-wrap').hidden = false; ttsLoadFailed('TTS worker crashed: ' + e.message) }
}
async function loadModel() {
  if (!SpeechRecognition) { $('status').textContent = 'Web Speech API not supported in this browser (use Chrome/Edge)'; $('speak-btn').disabled = true; return }
  setState('mic_only'); $('progress-wrap').hidden = false
  try { const r = await sendWorker({ type: 'load' }); modelReady = true; modelLoadDone = true; $('progress-wrap').hidden = true; $('persona-btn').disabled = false; setState('idle'); if (r?.device) $('status').textContent = `Ready — click Speak (${r.device})` }
  catch (err) { modelLoadDone = true; $('progress-wrap').hidden = true; $('status').textContent = `Model load failed: ${err.message} — mic still available` }
  ttsLoading = true; ttsWorker.postMessage({ type: 'load' })
}
async function speak(text) {
  if (ttsLoading && !ttsReady) $('status').textContent = 'TTS loading… (first run takes ~1 min)'
  const ready = await waitForTTS()
  if (!ready) { $('status').textContent = 'TTS unavailable'; return }
  ttsChunks = []
  await new Promise((resolve, reject) => {
    ttsChunkResolve = resolve; ttsChunkReject = reject
    ttsWorker.postMessage({ type: 'generate', data: { text, voice: $('voice-select').value } })
  })
}
function startRecognition() {
  return new Promise((resolve, reject) => {
    const rec = new SpeechRecognition(); recognition = rec; rec.lang = 'en-US'; rec.interimResults = true; rec.continuous = false
    let final = ''; rec.onresult = (e) => { final = Array.from(e.results).map(r => r[0].transcript).join(''); $('status').textContent = `Listening… ${final}` }
    rec.onend = () => resolve(final.trim()); rec.onerror = (e) => reject(new Error(e.error)); rec.start()
  })
}
$('speak-btn').addEventListener('click', async () => {
  if (appState === 'generating' || appState === 'speaking' || appState === 'recording') return
  setState('recording')
  let transcript = ''
  try { transcript = await startRecognition() } catch (err) { $('status').textContent = `Mic error: ${err.message}`; setState(modelReady ? 'idle' : (modelLoadDone ? 'idle' : 'mic_only')); return }
  if (!transcript) { setState(modelReady ? 'idle' : (modelLoadDone ? 'idle' : 'mic_only')); return }
  addBubble('user', transcript)
  if (!modelReady) {
    if (modelLoadDone) { setState('idle'); return }
    $('status').textContent = 'Model loading — waiting…'
    await new Promise(resolve => {
      const check = setInterval(() => { if (modelReady || modelLoadDone) { clearInterval(check); resolve() } }, 500)
    })
    if (!modelReady) { setState('idle'); return }
  }
  history.push({ role: 'user', content: transcript })
  setState('generating'); addBubble('assistant', '')
  setVRMPaused(true)
  try {
    let messages, genConfig
    if (personaDesc) {
      messages = [{ role: 'system', content: `You are ${personaDesc}. Stay in character as this specific person. Respond naturally and briefly.` }, ...personaHistory, ...history]; genConfig = { maxNewTokens: 120, temperature: 0.8 }
    } else {
      messages = [{ role: 'system', content: 'Reply in 1-2 sentences. Be concise. No lists.' }, ...history]; genConfig = { maxNewTokens: 40, temperature: 0.7 }
    }
    const { text } = await sendWorker({ type: 'generate', messages, config: genConfig })
    const cleaned = text.trim()
    history.push({ role: 'assistant', content: cleaned })
    const last = $('chat').querySelector('.bubble.assistant:last-child'); if (last) last.textContent = cleaned
    setState('speaking')
    setVRMPaused(false)
    await speak(cleaned)
  } catch (err) { setVRMPaused(false); $('status').textContent = `Error: ${err.message}` }
  setState('idle')
})
$('sheet-mic-btn').addEventListener('click', async () => {
  if (appState === 'generating' || appState === 'speaking' || appState === 'recording') return
  if (!SpeechRecognition) { $('status').textContent = 'Speech recognition not supported'; return }
  const btn = $('sheet-mic-btn'); btn.classList.add('recording'); const prevStatus = $('status').textContent; $('status').textContent = 'Listening…'
  let transcript = ''; try { transcript = await startRecognition() } catch (err) { $('status').textContent = `Mic error: ${err.message}`; btn.classList.remove('recording'); return }
  btn.classList.remove('recording'); if (!transcript) { $('status').textContent = prevStatus; return }
  $('character-sheet').value = transcript
  personaHistory = []; personaPrefill = null; personaDesc = ''; $('persona-btn').textContent = 'Generate Persona'
  $('status').textContent = prevStatus
})
const PERSONA_QUESTIONS = [
  'Who are you and what do you do around here?',
  'What are you working on right now?',
  'How long have you been doing this?',
  'What are you proud of?',
  'What do you want?',
  'What do you dislike?',
  'What happens if I cause trouble here?'
]
async function buildPersonaHistory(desc) {
  const sys = `You are ${desc}. You live and work in a game world. Speak only as this character would — use their voice, their concerns, their place in the world. Use plain short sentences. Never give advice. Never be abstract. Never sound like an assistant. Say what this character would actually say.`
  const turns = []
  setVRMPaused(true)
  for (const q of PERSONA_QUESTIONS) {
    $('persona-btn').textContent = `Shaping character… (${turns.length / 2 + 1}/${PERSONA_QUESTIONS.length})`
    const { text } = await sendWorker({ type: 'generate', messages: [{ role: 'system', content: sys }, ...turns, { role: 'user', content: q }], config: { maxNewTokens: 35, temperature: 0.8, repetitionPenalty: 1.3 } })
    const reply = text.trim().split('\n')[0].trim()
    turns.push({ role: 'user', content: q }, { role: 'assistant', content: reply || '...' })
  }
  setVRMPaused(false)
  return turns
}
$('persona-btn').addEventListener('click', async () => {
  const sheet = $('character-sheet').value.trim()
  if (!sheet) { $('status').textContent = 'Enter a character sheet first'; return }
  if (!modelReady) { $('status').textContent = modelLoadDone ? 'Model unavailable' : 'Model still loading'; return }
  $('persona-btn').disabled = true
  const desc = sheet.replace(/^(i am|i'm|name:|character:)\s*/i, '').trim()
  personaDesc = desc; history.length = 0
  personaHistory = [{ role: 'user', content: `you are ${sheet}` }, { role: 'assistant', content: 'ok' }, ...await buildPersonaHistory(desc)]
  await sendWorker({ type: 'reset' })
  personaPrefill = null
  console.log('[Persona] Active:', desc, personaHistory)
  $('persona-btn').textContent = `Character locked (${personaHistory.length / 2} turns)`
  $('persona-btn').disabled = false
})
$('voice-select').addEventListener('change', () => { ttsWorker.postMessage({ type: 'load_voice', voice: $('voice-select').value }) })
loadModel()
initVRM($('vrm-canvas')).catch(err => { console.warn('[VRM] Failed to load:', err.message); $('vrm-canvas').style.display = 'none' })
window.__app = { sendWorker, getPersonaHistory: () => personaHistory, setPersonaHistory: (h) => { personaHistory = h }, getPersonaPrefill: () => personaPrefill, setPersonaPrefill: (p) => { personaPrefill = p }, getPersonaDesc: () => personaDesc, getHistory: () => history, clearHistory: () => { history.length = 0 }, isModelReady: () => modelReady, isTtsReady: () => ttsReady, speak }
