import { initVRM, setVRMPaused } from './vrm-viewer.js'
import { startTTS, isTtsReady, onVoiceChange, speak } from './app-tts.js?v=4'
import { buildPersonaHistory } from './app-persona.js'

const worker = new Worker('./worker.js?v=65', { type: 'module' })
const SpeechRecognition = window.SpeechRecognition ?? window.webkitSpeechRecognition
const $ = (id) => document.getElementById(id)
const history = []
let pendingResolvers = {}, msgId = 0, appState = 'loading'
let modelReady = false, modelLoadDone = false
let personaHistory = [], personaPrefill = null, personaDesc = '', recognition = null

function nextId() { return ++msgId }
function setState(s) {
	appState = s
	$('status').textContent = { loading: 'Loading model…', idle: 'Ready — click Speak', recording: 'Listening…', generating: 'Thinking…', speaking: 'Speaking…', mic_only: 'Model loading — mic ready' }[s] ?? s
	$('speak-btn').disabled = s === 'generating' || s === 'speaking' || s === 'no_mic'
	$('speak-btn').classList.toggle('recording', s === 'recording')
}
function addBubble(role, text) {
	const div = Object.assign(document.createElement('div'), { className: `bubble ${role}`, textContent: text })
	$('chat').appendChild(div); $('chat').scrollTop = $('chat').scrollHeight
	return div
}
function sendWorker(msg) {
	return new Promise((resolve, reject) => {
		const id = nextId(); pendingResolvers[id] = { resolve, reject }
		worker.postMessage({ ...msg, id })
	})
}
worker.onmessage = (e) => {
	const { type, id, progress, message, token } = e.data
	if (type === 'progress') {
		if (!modelReady) {
			$('progress-wrap').hidden = false
			$('progress-bar').style.width = `${Math.round(progress?.progress ?? 0)}%`
			$('progress-text').textContent = progress?.file || 'Loading model…'
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
	const r = pendingResolvers[id]; if (!r) return
	delete pendingResolvers[id]
	if (type === 'error') r.reject(new Error(message)); else r.resolve(e.data)
}
worker.onerror = (e) => {
	const msg = [e.message, e.filename, e.lineno, e.colno, String(e.error)].filter(Boolean).join(' | ') || 'Worker crashed'
	Object.values(pendingResolvers).forEach(r => r.reject(new Error(msg)))
	Object.keys(pendingResolvers).forEach(k => delete pendingResolvers[k])
	$('progress-wrap').hidden = true
	$('status').textContent = `Model load failed: ${msg} — mic still available`
}
function startRecognition() {
	return new Promise((resolve, reject) => {
		const rec = new SpeechRecognition(); recognition = rec; rec.lang = 'en-US'; rec.interimResults = true; rec.continuous = false
		let final = ''; rec.onresult = (e) => { final = Array.from(e.results).map(r => r[0].transcript).join(''); $('status').textContent = `Listening… ${final}` }
		rec.onend = () => resolve(final.trim()); rec.onerror = (e) => reject(new Error(e.error)); rec.start()
	})
}
async function loadModel() {
	if (!SpeechRecognition) { $('status').textContent = 'Web Speech API not supported (use Chrome/Edge)'; $('speak-btn').disabled = true; return }
	setState('mic_only'); $('progress-wrap').hidden = false
	try {
		const r = await sendWorker({ type: 'load' })
		modelReady = true; modelLoadDone = true; $('progress-wrap').hidden = true; $('persona-btn').disabled = false
		setState('idle'); if (r?.device) $('status').textContent = `Ready — click Speak (${r.device})`
	} catch (err) { modelLoadDone = true; $('progress-wrap').hidden = true; $('status').textContent = `Model load failed: ${err.message} — mic still available` }
	startTTS()
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
		await new Promise(resolve => { const check = setInterval(() => { if (modelReady || modelLoadDone) { clearInterval(check); resolve() } }, 500) })
		if (!modelReady) { setState('idle'); return }
	}
	history.push({ role: 'user', content: transcript })
	setState('generating'); addBubble('assistant', ''); setVRMPaused(true)
	try {
		let messages, genConfig
		if (personaDesc) {
			messages = [{ role: 'system', content: `You are ${personaDesc}. Stay in character. Give the shortest, funniest possible response. One sentence max. Be witty.` }, ...personaHistory, ...history]; genConfig = { maxNewTokens: 120, temperature: 0.9 }
		} else {
			messages = [{ role: 'system', content: 'Reply in one sentence. Be as short and funny as possible.' }, ...history]; genConfig = { maxNewTokens: 40, temperature: 0.9 }
		}
		const { text } = await sendWorker({ type: 'generate', messages, config: genConfig })
		const cleaned = text.trim(); history.push({ role: 'assistant', content: cleaned })
		const last = $('chat').querySelector('.bubble.assistant:last-child'); if (last) last.textContent = cleaned
		setState('speaking'); setVRMPaused(false); await speak(cleaned)
	} catch (err) { setVRMPaused(false); $('status').textContent = `Error: ${err.message}` }
	setState('idle')
})
$('sheet-mic-btn').addEventListener('click', async () => {
	if (appState === 'generating' || appState === 'speaking' || appState === 'recording') return
	if (!SpeechRecognition) { $('status').textContent = 'Speech recognition not supported'; return }
	const btn = $('sheet-mic-btn'); btn.classList.add('recording'); const prev = $('status').textContent; $('status').textContent = 'Listening…'
	let transcript = ''; try { transcript = await startRecognition() } catch (err) { $('status').textContent = `Mic error: ${err.message}`; btn.classList.remove('recording'); return }
	btn.classList.remove('recording'); if (!transcript) { $('status').textContent = prev; return }
	$('character-sheet').value = transcript
	personaHistory = []; personaPrefill = null; personaDesc = ''; $('persona-btn').textContent = 'Generate Persona'
	$('status').textContent = prev
})
$('persona-btn').addEventListener('click', async () => {
	const sheet = $('character-sheet').value.trim()
	if (!sheet) { $('status').textContent = 'Enter a character sheet first'; return }
	if (!modelReady) { $('status').textContent = modelLoadDone ? 'Model unavailable' : 'Model still loading'; return }
	$('persona-btn').disabled = true
	const desc = sheet.replace(/^(i am|i'm|name:|character:)\s*/i, '').trim()
	personaDesc = desc; history.length = 0
	personaHistory = [{ role: 'user', content: `you are ${sheet}` }, { role: 'assistant', content: 'ok' }, ...await buildPersonaHistory(desc, sendWorker)]
	await sendWorker({ type: 'reset' }); personaPrefill = null
	$('persona-btn').textContent = `Character locked (${personaHistory.length / 2} turns)`
	$('persona-btn').disabled = false
})
$('voice-select').addEventListener('change', () => onVoiceChange($('voice-select').value))
loadModel()
initVRM($('vrm-canvas')).catch(err => { console.warn('[VRM] Failed:', err.message); $('vrm-canvas').style.display = 'none' })
window.__app = { sendWorker, getPersonaHistory: () => personaHistory, setPersonaHistory: (h) => { personaHistory = h }, getPersonaPrefill: () => personaPrefill, setPersonaPrefill: (p) => { personaPrefill = p }, getPersonaDesc: () => personaDesc, getHistory: () => history, clearHistory: () => { history.length = 0 }, isModelReady: () => modelReady, isTtsReady, speak }
