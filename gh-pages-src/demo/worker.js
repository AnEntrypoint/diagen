import { CreateMLCEngine } from './web-llm.js'

const MODEL = 'Llama-3.2-1B-Instruct-q4f16_1-MLC'
let engine = null, loading = false, loadError = null

self.onmessage = async (e) => {
	const { type, id } = e.data

	if (type === 'load') {
		if (engine) { self.postMessage({ type: 'loaded', id, device: 'webgpu' }); return }
		if (loading) return
		loading = true
		try {
			engine = await CreateMLCEngine(MODEL, {
				initProgressCallback: (p) => self.postMessage({
					type: 'progress',
					progress: { progress: Math.round((p.progress ?? 0) * 100), file: p.text ?? '' }
				}),
			})
			loading = false
			self.postMessage({ type: 'loaded', id, device: 'webgpu' })
		} catch (err) {
			loading = false; loadError = err.message
			self.postMessage({ type: 'error', message: err.message, id })
		}
		return
	}

	if (type === 'reset') {
		if (engine) await engine.resetChat().catch(() => {})
		self.postMessage({ type: 'reset', id })
		return
	}

	if (type === 'generate') {
		if (!engine) { self.postMessage({ type: 'error', message: loadError ?? 'Model not loaded', id }); return }
		const { messages, config = {} } = e.data
		try {
			const tokens = []
			const stream = await engine.chat.completions.create({
				messages,
				stream: true,
				max_tokens: config.maxNewTokens ?? 300,
				temperature: config.temperature ?? 0.7,
				frequency_penalty: config.repetitionPenalty ? config.repetitionPenalty - 1 : 0,
			})
			for await (const chunk of stream) {
				const token = chunk.choices[0]?.delta?.content ?? ''
				if (token) { tokens.push(token); self.postMessage({ type: 'token', token, id }) }
			}
			self.postMessage({ type: 'result', text: tokens.join(''), id })
		} catch (err) {
			self.postMessage({ type: 'error', message: err.message, id })
		}
		return
	}
}
