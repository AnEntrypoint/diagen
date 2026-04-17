const OLLAMA_URL = process.env.OLLAMA_URL || 'http://localhost:11434'
const MODEL = process.env.OLLAMA_MODEL || 'llama3.2:1b'

export async function generate(prompt, system = 'You are a helpful assistant. Be concise.', signal) {
  const t0 = Date.now()
  try {
    const res = await fetch(`${OLLAMA_URL}/api/generate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model: MODEL,
        prompt: `${system}\n\n${prompt}`,
        raw: true,
        stream: false,
        options: { temperature: 0.9, top_p: 0.92, num_predict: 180, repeat_penalty: 1.25, stop: ['\n\n'] },
      }),
      signal,
    })

    if (!res.ok) {
      const text = await res.text()
      throw new Error(`Ollama error ${res.status}: ${text}`)
    }

    const data = await res.json()
    const dur = Date.now() - t0
    console.log(`[ollama] gen ${dur}ms model=${MODEL} eval=${data.eval_count || '?'}t`)
    return data.response
  } catch (err) {
    if (err.name === 'AbortError') { console.log(`[ollama] aborted after ${Date.now()-t0}ms`); throw err }
    console.error(`[ollama] error after ${Date.now()-t0}ms:`, err.message)
    throw err
  }
}

export async function generateStream(prompt, system = 'You are a helpful assistant. Be concise.') {
  const res = await fetch(`${OLLAMA_URL}/api/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      model: MODEL,
      messages: [
        { role: 'system', content: system },
        { role: 'user', content: prompt },
      ],
      stream: true,
    }),
  })

  if (!res.ok) {
    const text = await res.text()
    throw new Error(`Ollama error ${res.status}: ${text}`)
  }

  return res.body
}

export async function isAvailable() {
  try {
    const res = await fetch(`${OLLAMA_URL}/api/tags`, { signal: AbortSignal.timeout(3000) })
    if (!res.ok) return false
    const data = await res.json()
    return data.models?.some(m => m.name === MODEL || m.name.startsWith(MODEL.split(':')[0]))
  } catch {
    return false
  }
}

export default { generate, generateStream, isAvailable }
