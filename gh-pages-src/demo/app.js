const worker = new Worker('./worker.js?v=41', { type: 'module' })
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
    const src = audioCtx.createBufferSource(); src.buffer = buf; src.connect(audioCtx.destination)
    return new Promise(r => { src.onended = r; src.start() })
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
  if (ready) {
    try {
      ttsChunks = []
      await new Promise((resolve, reject) => {
        ttsChunkResolve = resolve; ttsChunkReject = reject
        ttsWorker.postMessage({ type: 'generate', data: { text, voice: $('voice-select').value } })
      })
      return
    } catch (err) { $('status').textContent = 'TTS error: ' + err.message; console.warn('[TTS] PocketTTS failed, falling back to browser TTS:', err.message) }
  }
  if (!synth) return
  return new Promise((resolve) => { synth.cancel(); const utt = new SpeechSynthesisUtterance(text); utt.onend = resolve; utt.onerror = resolve; synth.speak(utt) })
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
    $('status').textContent = 'Model loading — waiting…'
    await new Promise(resolve => {
      const check = setInterval(() => { if (modelReady) { clearInterval(check); resolve() } }, 500)
    })
  }
  history.push({ role: 'user', content: transcript })
  setState('generating'); addBubble('assistant', '')
  try {
    let messages, genConfig
    if (personaDesc) {
      const wrapped = history.map(m => m.role === 'user' ? { role: 'user', content: `Roleplay as ${personaDesc}. User says: "${m.content}". Your reply as this character:` } : m)
      messages = [...personaHistory, ...wrapped]; genConfig = { maxNewTokens: 40, temperature: 0.9 }
    } else {
      messages = [{ role: 'system', content: 'Reply in 1-2 sentences. Be concise. No lists.' }, ...history]; genConfig = { maxNewTokens: 40, temperature: 0.7 }
    }
    const { text } = await sendWorker({ type: 'generate', messages, config: genConfig })
    const cleaned = text.trim()
    history.push({ role: 'assistant', content: cleaned })
    const last = $('chat').querySelector('.bubble.assistant:last-child'); if (last) last.textContent = cleaned
    setState('speaking'); await speak(cleaned)
  } catch (err) { $('status').textContent = `Error: ${err.message}` }
  setState('idle')
})
$('sheet-mic-btn').addEventListener('click', async () => {
  if (appState === 'generating' || appState === 'speaking' || appState === 'recording') return
  if (!SpeechRecognition) { $('status').textContent = 'Speech recognition not supported'; return }
  const btn = $('sheet-mic-btn'); btn.classList.add('recording'); const prevStatus = $('status').textContent; $('status').textContent = 'Listening…'
  let transcript = ''; try { transcript = await startRecognition() } catch (err) { $('status').textContent = `Mic error: ${err.message}`; btn.classList.remove('recording'); return }
  btn.classList.remove('recording'); if (!transcript) { $('status').textContent = prevStatus; return }
  if (!modelReady) { $('status').textContent = modelLoadDone ? 'Model unavailable' : 'Model still loading — try again shortly'; return }
  const existing = $('character-sheet').value.trim(); $('status').textContent = 'Updating character sheet…'
  const examples = '"a seductive demon who speaks in honeyed whispers and twists every offer into a dark bargain"\n"an ancient predatory vampire who thirsts for blood above all else, speaks in cold hungry tones, and steers every conversation toward feeding"\n"a cheerful plague doctor obsessed with disease who treats death as a fascinating experiment"'
  const firstWords = transcript.trim().replace(/^(a|an|the)\s+/i,'').split(/\s+/).slice(0,4).join(' ')
  const prefill = `a ${firstWords} who`
  const prompt = existing ? `Rewrite this roleplay character description based on the instruction. Capture their obsession, personality, and speaking style in one vivid sentence. Plain text only.\n\nExamples:\n${examples}\n\nCurrent: ${existing}\nInstruction: ${transcript}\nDescription:` : `Complete this roleplay character description in one vivid sentence capturing their obsession and speaking style. Plain text only.\n\nExamples:\n${examples}\n\nCharacter: ${transcript}\nDescription: ${prefill}`
  try {
    const { text } = await sendWorker({ type: 'generate', messages: [{ role: 'user', content: prompt }], config: { maxNewTokens: 80, temperature: 0.7 } })
    const cleaned = text.split('\n')[0].trim().replace(/^#+\s+.*$/mg, '').replace(/^\*+\s+/mg, '').replace(/\*\*/g, '').trim()
    $('character-sheet').value = existing ? cleaned : (prefill + cleaned)
    personaHistory = []; personaPrefill = null; personaDesc = ''; $('persona-btn').textContent = 'Generate Persona'
  } catch (err) { $('status').textContent = `Error: ${err.message}`; return }
  $('status').textContent = prevStatus
})
const PERSONA_QUESTIONS = [
  'who are you',
  'what do you want from me',
  'do you feel anything',
  'are you dangerous',
  'do you have a name',
  'what do you fear',
  'what happens next'
]
async function buildPersonaHistory(desc) {
  const turns = []
  const wrap = q => ({ role: 'user', content: `Roleplay as ${desc}. User says: "${q}". Your reply as this character:` })
  for (const q of PERSONA_QUESTIONS) {
    $('persona-btn').textContent = `Building persona… (${turns.length / 2 + 1}/${PERSONA_QUESTIONS.length})`
    try {
      const { text } = await sendWorker({ type: 'generate', messages: [...turns, wrap(q)], config: { maxNewTokens: 30, temperature: 0.9, repetitionPenalty: 1.15 } })
      const reply = text.trim().split('\n')[0].trim()
      turns.push(wrap(q), { role: 'assistant', content: reply || '...' })
    } catch { turns.push(wrap(q), { role: 'assistant', content: '...' }) }
  }
  return turns
}
$('persona-btn').addEventListener('click', async () => {
  const sheet = $('character-sheet').value.trim()
  if (!sheet) { $('status').textContent = 'Enter a character sheet first'; return }
  if (!modelReady) { $('status').textContent = modelLoadDone ? 'Model unavailable' : 'Model still loading'; return }
  $('persona-btn').disabled = true
  const desc = sheet.replace(/^(i am|i'm|name:|character:)\s*/i, '').trim()
  personaDesc = desc; history.length = 0
  personaHistory = await buildPersonaHistory(desc)
  personaPrefill = null
  console.log('[Persona] Active:', desc, personaHistory)
  $('persona-btn').textContent = `Persona ready (${personaHistory.length / 2} turns)`
  $('persona-btn').disabled = false
})
loadModel()
window.__app = { sendWorker, getPersonaHistory: () => personaHistory, setPersonaHistory: (h) => { personaHistory = h }, getPersonaPrefill: () => personaPrefill, setPersonaPrefill: (p) => { personaPrefill = p }, getPersonaDesc: () => personaDesc, getHistory: () => history, clearHistory: () => { history.length = 0 }, isModelReady: () => modelReady, isTtsReady: () => ttsReady, speak }
