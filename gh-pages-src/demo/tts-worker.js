import init, { Model } from './wasm_pocket_tts.js'

const HF_BASE = 'https://huggingface.co/kyutai/pocket-tts-without-voice-cloning/resolve/main'
const MODEL_URL = `${HF_BASE}/tts_b6369a24.safetensors`
const TOKENIZER_URL = `${HF_BASE}/tokenizer.model`
const VOICE_NAMES = ['alba', 'marius', 'javert', 'fantine', 'cosette', 'eponine', 'azelma']
const CUSTOM_VOICE_NAMES = ['cleetus', 'vampire']
const DEFAULT_VOICE = 'alba'
const CACHE_NAME = 'tts-wasm-cache'

const wasmModulePromise = WebAssembly.compileStreaming(fetch('./wasm_pocket_tts_bg.wasm'))

let model = null, tokenizer = null, voiceIndexMap = {}, customVoiceIndexMap = {}

function decodeSentencepieceModel(buffer) {
  let pos = 0
  function readVarint() {
    let result = 0, shift = 0
    while (pos < buffer.length) {
      const b = buffer[pos++]
      result |= (b & 0x7f) << shift; shift += 7
      if ((b & 0x80) === 0) return result
    }
    return result
  }
  function readVarFrom(buf, p) {
    let result = 0, shift = 0
    while (p < buf.length) {
      const b = buf[p++]
      result |= (b & 0x7f) << shift; shift += 7
      if ((b & 0x80) === 0) return { val: result, pos: p }
    }
    return { val: result, pos: p }
  }
  function decodePiece(data) {
    let pPos = 0, piece = '', score = 0, type = 1
    const pView = new DataView(data.buffer, data.byteOffset, data.byteLength)
    while (pPos < data.length) {
      const key = readVarFrom(data, pPos); pPos = key.pos
      const fieldNum = key.val >>> 3, wireType = key.val & 0x7
      if (fieldNum === 1 && wireType === 2) {
        const len = readVarFrom(data, pPos); pPos = len.pos
        piece = new TextDecoder().decode(data.slice(pPos, pPos + len.val)); pPos += len.val
      } else if (fieldNum === 2 && wireType === 5) { score = pView.getFloat32(pPos, true); pPos += 4 }
      else if (fieldNum === 3 && wireType === 0) { const v = readVarFrom(data, pPos); type = v.val; pPos = v.pos }
      else {
        if (wireType === 0) { const v = readVarFrom(data, pPos); pPos = v.pos }
        else if (wireType === 1) pPos += 8
        else if (wireType === 2) { const len = readVarFrom(data, pPos); pPos = len.pos + len.val }
        else if (wireType === 5) pPos += 4
        else break
      }
    }
    return { piece, score, type }
  }
  const pieces = []
  while (pos < buffer.length) {
    const key = readVarint(), fieldNum = key >>> 3, wireType = key & 0x7
    if (fieldNum === 1 && wireType === 2) {
      const len = readVarint(), data = buffer.slice(pos, pos + len); pos += len
      pieces.push(decodePiece(data))
    } else {
      if (wireType === 0) readVarint()
      else if (wireType === 1) pos += 8
      else if (wireType === 2) { const len = readVarint(); pos += len }
      else if (wireType === 5) pos += 4
      else break
    }
  }
  return pieces
}

class UnigramTokenizer {
  constructor(pieces) {
    this.vocab = new Map(); this.unkId = 0
    for (let i = 0; i < pieces.length; i++) {
      const p = pieces[i]
      if (p.type === 2) this.unkId = i
      if (p.type === 1 || p.type === 4 || p.type === 6) this.vocab.set(p.piece, { id: i, score: p.score })
    }
  }
  encode(text) {
    const s = '\u2581' + text.replace(/ /g, '\u2581'), n = s.length
    const best = new Array(n + 1).fill(null).map(() => ({ score: -Infinity, len: 0, id: -1 }))
    best[0].score = 0
    for (let i = 0; i < n; i++) {
      if (best[i].score === -Infinity) continue
      for (let len = 1; len <= n - i && len <= 64; len++) {
        const e = this.vocab.get(s.substring(i, i + len))
        if (e) { const ns = best[i].score + e.score; if (ns > best[i+len].score) best[i+len] = { score: ns, len, id: e.id } }
      }
      if (best[i+1].score === -Infinity) {
        const ch = s.charCodeAt(i), bs = `<0x${ch.toString(16).toUpperCase().padStart(2,'0')}>`
        const be = this.vocab.get(bs)
        best[i+1] = { score: best[i].score + (be ? be.score : -100), len: 1, id: be ? be.id : this.unkId }
      }
    }
    const ids = []; let p = n
    while (p > 0) { ids.push(best[p].id); p -= best[p].len }
    ids.reverse(); return new Uint32Array(ids)
  }
}

async function fetchBufWithCache(url, cacheKey) {
  try {
    const cache = await caches.open(CACHE_NAME)
    const cached = await cache.match(cacheKey)
    if (cached) {
      const buf = await cached.arrayBuffer()
      return new Uint8Array(buf)
    }
    const resp = await fetch(url)
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`)
    const buf = await resp.arrayBuffer()
    const cached_resp = new Response(buf, { headers: { 'Content-Type': 'application/octet-stream' } })
    await cache.put(cacheKey, cached_resp)
    return new Uint8Array(buf)
  } catch (e) {
    self.postMessage({ type: 'error', error: 'Failed to fetch ' + cacheKey + ': ' + e.message })
    throw e
  }
}


async function handleLoad() {
  self.postMessage({ type: 'status', status: 'Initializing WASM...' })
  const wasmModule = await wasmModulePromise
  await init(wasmModule)

  self.postMessage({ type: 'status', status: 'Loading tokenizer (may download ~60KB)...' })
  const tokData = await fetchBufWithCache(TOKENIZER_URL, 'tts-tokenizer')
  tokenizer = new UnigramTokenizer(decodeSentencepieceModel(tokData))

  self.postMessage({ type: 'status', status: 'Loading model (may download ~200MB, first time only)...' })
  const modelWeights = await fetchBufWithCache(MODEL_URL, 'tts-model')
  model = new Model(modelWeights)

  voiceIndexMap = {}
  customVoiceIndexMap = {}
  self.postMessage({ type: 'voices_loaded', voices: [...VOICE_NAMES, ...CUSTOM_VOICE_NAMES], defaultVoice: DEFAULT_VOICE })
  self.postMessage({ type: 'loaded' })
}

async function handleLoadVoice(name) {
  if (voiceIndexMap[name] != null || customVoiceIndexMap[name] != null) return
  self.postMessage({ type: 'status', status: `Loading voice: ${name}` })
  if (CUSTOM_VOICE_NAMES.includes(name)) {
    const voiceData = await fetchBufWithCache(`./voices/${name}.safetensors`, `tts-voice-${name}`)
    customVoiceIndexMap[name] = model.add_voice(voiceData)
  } else {
    const voiceData = await fetchBufWithCache(`${HF_BASE}/embeddings_v2/${name}.safetensors`, `tts-voice-${name}`)
    voiceIndexMap[name] = model.add_voice(voiceData)
  }
  self.postMessage({ type: 'voice_ready', voice: name })
}

async function handleGenerate(text, voiceName) {
  if (voiceIndexMap[voiceName] == null && customVoiceIndexMap[voiceName] == null) await handleLoadVoice(voiceName)
  const idx = voiceIndexMap[voiceName] ?? customVoiceIndexMap[voiceName] ?? voiceIndexMap[DEFAULT_VOICE]
  const [processedText, framesAfterEos] = model.prepare_text(text)
  const tokenIds = tokenizer.encode(processedText)
  model.start_generation(idx, tokenIds, framesAfterEos, 0.8)
  while (true) {
    const chunk = model.generation_step()
    if (!chunk) break
    self.postMessage({ type: 'audio_chunk', data: chunk.buffer }, [chunk.buffer])
    await new Promise(r => setTimeout(r, 0))
  }
  self.postMessage({ type: 'stream_ended' })
}

self.onmessage = async (e) => {
  const { type } = e.data
  try {
    if (type === 'load') await handleLoad()
    else if (type === 'load_voice') await handleLoadVoice(e.data.voice)
    else if (type === 'generate') await handleGenerate(e.data.data?.text || e.data.text, e.data.data?.voice || e.data.voice || DEFAULT_VOICE)
  } catch (err) {
    self.postMessage({ type: 'error', error: err.message })
  }
}
