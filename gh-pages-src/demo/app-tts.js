import { setMouthOpen } from './vrm-viewer.js'

const ttsWorker = new Worker('./tts-worker.js', { type: 'module' })
let ttsReady = false, ttsLoading = false
let ttsReadyResolvers = [], ttsChunkResolve = null, ttsChunkReject = null, ttsChunks = []
let audioCtx = null

const $ = (id) => document.getElementById(id)

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
export async function speak(text) {
	if (ttsLoading && !ttsReady) $('status').textContent = 'TTS loading… (first run takes ~1 min)'
	const ready = await waitForTTS()
	if (!ready) { $('status').textContent = 'TTS unavailable'; return }
	ttsChunks = []
	await new Promise((resolve, reject) => {
		ttsChunkResolve = resolve; ttsChunkReject = reject
		ttsWorker.postMessage({ type: 'generate', data: { text, voice: $('voice-select').value } })
	})
}
