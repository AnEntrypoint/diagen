import { CreateMLCEngine } from '/node_modules/@mlc-ai/web-llm/lib/index.js'

const MODEL = 'Llama-3.2-1B-Instruct-q4f16_1-MLC'
let engine = null

async function load() {
	engine = await CreateMLCEngine(MODEL, {
		initProgressCallback: (p) => self.postMessage({ type: 'progress', text: p.text, progress: p.progress }),
	})
	self.postMessage({ type: 'loaded' })
}

async function generate(prompt, system) {
	const chunks = await engine.chat.completions.create({
		messages: [
			{ role: 'system', content: system },
			{ role: 'user', content: prompt },
		],
		stream: true,
	})
	let full = ''
	for await (const chunk of chunks) {
		const text = chunk.choices[0]?.delta?.content || ''
		if (text) { full += text; self.postMessage({ type: 'chunk', text }) }
	}
	self.postMessage({ type: 'done', text: full })
}

self.onmessage = async (e) => {
	const { type, prompt, system } = e.data
	try {
		if (type === 'load') await load()
		else if (type === 'generate') await generate(prompt, system || 'You are a helpful assistant. Be concise.')
	} catch (err) {
		self.postMessage({ type: 'error', error: err.message })
	}
}
