import { setVRMPaused } from './vrm-viewer.js'

const $ = (id) => document.getElementById(id)
const PERSONA_QUESTIONS = [
	'Who are you and what do you do around here?',
	'What are you working on right now?',
	'How long have you been doing this?',
	'What are you proud of?',
	'What do you want?',
	'What do you dislike?',
	'What happens if I cause trouble here?',
]

export async function buildPersonaHistory(desc, sendWorker) {
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
