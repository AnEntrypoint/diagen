import { generate as generateLLM, isAvailable as isLLMAvailable, buildGrammar } from './llm-llamacpp.js'
import { synthesizeStream } from './qwen3-tts-bridge.js'
import { resampleAudio } from './server-utils.mjs'

const SAMPLE_RATE_DISCORD = 48000
const SAMPLE_RATE_TTS_FALLBACK = 24000
const DEBOUNCE_MS = Number(process.env.GATE_DEBOUNCE_MS || 1000)
const STAGE_TIMEOUT = {
  GATING: Number(process.env.GATE_TIMEOUT_GATING_MS || 5000),
  ANSWERING: Number(process.env.GATE_TIMEOUT_ANSWER_MS || 15000),
  SPEAKING: Number(process.env.GATE_TIMEOUT_SPEAKING_MS || 30000),
}
const MAX_RESPONSE_CHARS = 280
const MAX_HISTORY = 12

const GATE_PROMPT = [
  'You decide whether the bot should speak now. Read the recent conversation. The user just stopped talking.',
  'Reply YES by default — the bot is conversational and should join in. Lean YES whenever the user said anything substantive, asked a question, addressed the bot, used the bot\'s name, or made a remark worth reacting to.',
  'Reply NO only when the user clearly addressed someone else by name, was obviously mid-sentence with no pause, said something trivial like a single filler word, or the bot already replied to this exact thing.',
  'Output only YES or NO.',
].join('\n')

let yesNoGrammar = null
async function getYesNoGrammar() {
  if (yesNoGrammar) return yesNoGrammar
  yesNoGrammar = await buildGrammar('root ::= "YES" | "NO"')
  return yesNoGrammar
}

const state = {
  name: 'LISTENING', enteredAt: Date.now(), debounceTimer: null, abort: null,
  lastWhisperAt: 0, lastDecision: null,
  audioSink: null, refPath: null, refText: null, characterPrompt: null,
  history: [], activeSpeakers: new Map(),
  metrics: { gateYes: 0, gateNo: 0, abortsByStage: { GATING: 0, ANSWERING: 0, SPEAKING: 0 }, timeouts: 0, spoken: 0 },
}

function setState(next, reason = '') {
  console.log(`[gate] ${state.name} → ${next}${reason ? ` (${reason})` : ''}`)
  state.name = next; state.enteredAt = Date.now()
}

function abortCurrent(reason) {
  if (!state.abort) return
  state.metrics.abortsByStage[state.name] = (state.metrics.abortsByStage[state.name] || 0) + 1
  try { state.abort.abort(reason) } catch {}
  state.abort = null
}

function snapHistory(role, text, username = null) {
  if (!text) return
  state.history.push({ role, username, text, timestamp: Date.now() })
  if (state.history.length > MAX_HISTORY) state.history.splice(0, state.history.length - MAX_HISTORY)
}

function buildContext() {
  return state.history.slice(-MAX_HISTORY).map(h =>
    h.role === 'user' ? `${h.username || 'user'}: "${h.text}"` : `bot: "${h.text}"`
  ).join('\n')
}

function armDebounce() {
  if (state.debounceTimer) clearTimeout(state.debounceTimer)
  state.debounceTimer = setTimeout(() => { state.debounceTimer = null; if (state.name === 'WAITING') runStage('GATING') }, DEBOUNCE_MS)
}

const onWhisperAbort = (reason) => () => { abortCurrent(reason); setState('WAITING', 'aborted by whisper'); armDebounce() }

const transitions = {
  LISTENING: { onWhisperWord: () => { setState('WAITING', 'first whisper word'); armDebounce() } },
  WAITING:   { onWhisperWord: () => armDebounce() },
  GATING:    { onWhisperWord: onWhisperAbort('whisper-mid-gate') },
  ANSWERING: { onWhisperWord: onWhisperAbort('whisper-mid-answer') },
  SPEAKING:  { onWhisperWord: onWhisperAbort('whisper-mid-speak') },
}

async function runStage(stage) {
  setState(stage)
  const abort = new AbortController()
  state.abort = abort
  const timer = setTimeout(() => { state.metrics.timeouts++; try { abort.abort('stage-timeout') } catch {} }, STAGE_TIMEOUT[stage])
  try {
    const handler = stageHandlers[stage]
    await handler(abort)
  } catch (err) {
    if (err?.name !== 'AbortError') console.error(`[gate] ${stage} error:`, err.message)
    if (state.name === stage) setState('LISTENING', `${stage}-err:${err?.name || 'x'}`)
  } finally {
    clearTimeout(timer)
    if (state.abort === abort) state.abort = null
  }
}

const stageHandlers = {
  GATING: async (abort) => {
    if (!(await isLLMAvailable())) { setState('LISTENING', 'LLM offline'); return }
    const grammar = await getYesNoGrammar()
    const raw = await generateLLM(`${buildContext()}\n\n${GATE_PROMPT}\n\nDecision:`, state.characterPrompt || undefined, abort.signal, { grammar, maxTokens: 4 })
    if (state.abort !== abort) return
    const decision = (raw || '').trim().toUpperCase().startsWith('Y') ? 'YES' : 'NO'
    state.lastDecision = { decision, at: Date.now() }
    state.metrics[decision === 'YES' ? 'gateYes' : 'gateNo']++
    console.log(`[gate] decision=${decision} raw="${(raw || '').slice(0, 20)}"`)
    if (decision === 'YES') runStage('ANSWERING')
    else setState('LISTENING', 'gate=NO')
  },
  ANSWERING: async (abort) => {
    const now = Date.now()
    const recent = [...state.activeSpeakers.values()].filter(s => now - s.lastWordAt < 5000)
    const multiHint = recent.length >= 2
      ? `\n\nMultiple people just spoke at the same time: ${recent.map(s => s.username).join(' and ')}. Address both in your one reply.`
      : ''
    const raw = await generateLLM(`${buildContext()}${multiHint}\n\nReply with the bot's next spoken turn. Keep it conversational and short.`, state.characterPrompt || undefined, abort.signal)
    if (state.abort !== abort) return
    const text = (raw || '').trim().slice(0, MAX_RESPONSE_CHARS)
    if (!text) { setState('LISTENING', 'empty answer'); return }
    state._pendingResponse = text
    runStage('SPEAKING')
  },
  SPEAKING: async (abort) => {
    const text = state._pendingResponse || ''
    state._pendingResponse = null
    if (!text) { setState('LISTENING', 'no text'); return }
    let chunksPlayed = 0
    const onChunk = (audio, sr) => {
      if (abort.signal.aborted || !state.audioSink) return
      state.audioSink(resampleAudio(audio, sr || SAMPLE_RATE_TTS_FALLBACK, SAMPLE_RATE_DISCORD), text)
      chunksPlayed++
    }
    try {
      await synthesizeStream(text, state.refPath, state.refText, onChunk, abort.signal)
    } finally {
      if (chunksPlayed > 0) {
        const words = text.split(/\s+/)
        const partial = abort.signal.aborted ? words.slice(0, Math.max(1, Math.floor(words.length * (chunksPlayed / (chunksPlayed + 2))))).join(' ') : text
        snapHistory('bot', partial)
        if (!abort.signal.aborted) state.metrics.spoken++
      }
      if (state.name === 'SPEAKING') setState('LISTENING', `done chunks=${chunksPlayed}`)
    }
  },
}

const MIN_WORD_CHARS = Number(process.env.GATE_MIN_WORD_CHARS || 3)

function stripSentinels(text) {
  return text.replace(/\[[^\]]*\]|\*[^*]*\*|\([^)]*\)/g, ' ').trim()
}

function isWordlessOrSentinel(text) {
  if (!text) return true
  const stripped = stripSentinels(text.trim())
  const alphanumCount = (stripped.match(/[a-zA-Z0-9]/g) || []).length
  return alphanumCount < MIN_WORD_CHARS
}

export function noteWhisperWord({ userId, username, text }) {
  if (isWordlessOrSentinel(text)) return
  state.lastWhisperAt = Date.now()
  state.activeSpeakers.set(userId, { username, lastWordAt: state.lastWhisperAt, lastText: text })
  const last = state.history[state.history.length - 1]
  if (last && last.role === 'user' && last.username === username) last.text = text
  else snapHistory('user', text, username)
  transitions[state.name]?.onWhisperWord?.()
}

export function setRefVoice(refPath, refText) { state.refPath = refPath; state.refText = refText }
export function setCharacterCardPrompt(prompt) { state.characterPrompt = prompt }
export function setAudioSink(fn) { state.audioSink = fn }
export function clearHistory() { state.history.length = 0; console.log('[gate] history cleared') }

export function getDebugSnapshot() {
  return {
    state: state.name,
    msInState: Date.now() - state.enteredAt,
    debounceArmed: Boolean(state.debounceTimer),
    msUntilTick: state.debounceTimer ? Math.max(0, DEBOUNCE_MS - (Date.now() - state.lastWhisperAt)) : null,
    activeAbortReason: state.abort ? 'in-flight' : null,
    lastDecision: state.lastDecision,
    history: state.history.slice(-10),
    activeSpeakers: [...state.activeSpeakers.entries()].map(([uid, v]) => ({ userId: uid, ...v })),
    metrics: state.metrics,
  }
}

export default { noteWhisperWord, setRefVoice, setCharacterCardPrompt, setAudioSink, clearHistory, getDebugSnapshot }
