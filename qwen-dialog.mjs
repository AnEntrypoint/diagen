import fs from 'fs'
import path from 'path'
import { pipeline } from 'stream/promises'
import { createWriteStream } from 'fs'
import { fileURLToPath } from 'url'
import ort from 'onnxruntime-node'

const __dirname = path.dirname(fileURLToPath(import.meta.url))
const HF_BASE = 'https://huggingface.co/onnx-community/Qwen2.5-0.5B-Instruct-ONNX/resolve/main'
const MODEL_DIR = path.join(__dirname, 'models', 'qwen')
const ONNX_FILE = 'onnx/model_q4f16.onnx'
const DL_FILES = ['config.json', 'generation_config.json', 'tokenizer.json', 'tokenizer_config.json', ONNX_FILE]
const EOS_IDS = new Set([151645, 151643])

async function dlFile(url, dest) {
  fs.mkdirSync(path.dirname(dest), { recursive: true })
  const res = await fetch(url, { signal: AbortSignal.timeout(600000), redirect: 'follow' })
  if (!res.ok) throw new Error(`HTTP ${res.status} for ${url}`)
  const ws = createWriteStream(dest + '.tmp')
  await pipeline(res.body, ws)
  fs.renameSync(dest + '.tmp', dest)
}

export async function downloadQwenModel() {
  for (const file of DL_FILES) {
    const dest = path.join(MODEL_DIR, file)
    if (fs.existsSync(dest) && fs.statSync(dest).size > 0) { console.log(`[qwen] ${file} cached`); continue }
    fs.mkdirSync(path.dirname(dest), { recursive: true })
    console.log(`[qwen] downloading ${file}...`)
    await dlFile(`${HF_BASE}/${file}`, dest)
    console.log(`[qwen] ${file} done (${(fs.statSync(dest).size / 1e6).toFixed(1)} MB)`)
  }
}

function buildTokenizer(tokJson) {
  const { model, added_tokens = [] } = tokJson
  const vocab = model.vocab
  const idToToken = Object.fromEntries(Object.entries(vocab).map(([t, id]) => [id, t]))
  for (const at of added_tokens) idToToken[at.id] = at.content
  const addedMap = Object.fromEntries(added_tokens.map(at => [at.content, at.id]))
  const addedList = added_tokens.map(at => at.content).sort((a, b) => b.length - a.length)
  const merges = model.merges.map(m => m.split(' '))

  const byteEnc = {}
  let n = 0
  for (let b = 0; b < 256; b++) {
    if ((b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174 && b <= 255)) byteEnc[b] = String.fromCharCode(b)
    else byteEnc[b] = String.fromCharCode(256 + n++)
  }
  const byteDec = Object.fromEntries(Object.entries(byteEnc).map(([b, c]) => [c, +b]))

  function bpe(word) {
    let w = [...word]
    while (w.length > 1) {
      let best = -1, bestPair = null
      for (let i = 0; i < w.length - 1; i++) {
        const idx = merges.findIndex(m => m[0] === w[i] && m[1] === w[i + 1])
        if (idx >= 0 && (best < 0 || idx < best)) { best = idx; bestPair = i }
      }
      if (best < 0) break
      w.splice(bestPair, 2, w[bestPair] + w[bestPair + 1])
    }
    return w
  }

  function tokenize(text) {
    const ids = []
    let rem = text
    while (rem.length > 0) {
      let hit = false
      for (const p of addedList) {
        if (rem.startsWith(p)) { ids.push(addedMap[p]); rem = rem.slice(p.length); hit = true; break }
      }
      if (hit) continue
      const m = rem.match(/^(\s*\S+|\s+)/)
      if (!m) break
      const chunk = m[0]
      rem = rem.slice(chunk.length)
      const enc = Array.from(new TextEncoder().encode(chunk)).map(b => byteEnc[b]).join('')
      for (const tok of bpe([...enc])) ids.push(vocab[tok] ?? 0)
    }
    return ids
  }

  function decode(ids) {
    const chars = ids.map(id => idToToken[id] ?? '').join('').split('')
    return new TextDecoder().decode(new Uint8Array(chars.map(c => byteDec[c] ?? c.charCodeAt(0))))
  }

  return { tokenize, decode }
}

let sess = null, tok = null, cfg = null, loadP = null

export async function loadQwenModel() {
  if (sess) return
  if (loadP) return loadP
  loadP = (async () => {
    cfg = JSON.parse(fs.readFileSync(path.join(MODEL_DIR, 'config.json'), 'utf8'))
    tok = buildTokenizer(JSON.parse(fs.readFileSync(path.join(MODEL_DIR, 'tokenizer.json'), 'utf8')))
    console.log('[qwen] loading ONNX session...')
    sess = await ort.InferenceSession.create(path.join(MODEL_DIR, ONNX_FILE), {
      executionProviders: ['cpu'], graphOptimizationLevel: 'all',
    })
    console.log('[qwen] ready, inputs:', sess.inputNames.slice(0, 3).join(', '), '...')
  })()
  return loadP
}

export async function generateDialog(prompt, { maxNewTokens = 200, system = 'You are a helpful dialog assistant. Be concise.' } = {}) {
  await loadQwenModel()
  const text = `<|im_start|>system\n${system}<|im_end|>\n<|im_start|>user\n${prompt}<|im_end|>\n<|im_start|>assistant\n`
  const ids = tok.tokenize(text)
  const numLayers = cfg.num_hidden_layers
  const numKvHeads = cfg.num_key_value_heads
  const headDim = Math.floor(cfg.hidden_size / cfg.num_attention_heads)
  const emptyKv = () => new ort.Tensor('float16', new Uint16Array(0), [1, numKvHeads, 0, headDim])

  const feeds = {
    input_ids: new ort.Tensor('int64', BigInt64Array.from(ids.map(BigInt)), [1, ids.length]),
    attention_mask: new ort.Tensor('int64', BigInt64Array.from(ids.map(() => 1n)), [1, ids.length]),
    position_ids: new ort.Tensor('int64', BigInt64Array.from(ids.map((_, i) => BigInt(i))), [1, ids.length]),
  }
  for (let i = 0; i < numLayers; i++) {
    feeds[`past_key_values.${i}.key`] = emptyKv()
    feeds[`past_key_values.${i}.value`] = emptyKv()
  }

  const out_ids = []
  let seqLen = ids.length
  for (let step = 0; step < maxNewTokens; step++) {
    const out = await sess.run(feeds)
    const logits = out.logits.data
    const vSize = out.logits.dims[2]
    const offset = (out.logits.dims[1] - 1) * vSize
    let maxV = -Infinity, nextId = 0
    for (let v = 0; v < vSize; v++) { if (logits[offset + v] > maxV) { maxV = logits[offset + v]; nextId = v } }
    if (EOS_IDS.has(nextId)) break
    out_ids.push(nextId)
    seqLen++
    feeds.input_ids = new ort.Tensor('int64', BigInt64Array.from([BigInt(nextId)]), [1, 1])
    feeds.attention_mask = new ort.Tensor('int64', BigInt64Array.from(new Array(seqLen).fill(1n)), [1, seqLen])
    feeds.position_ids = new ort.Tensor('int64', BigInt64Array.from([BigInt(seqLen - 1)]), [1, 1])
    for (let i = 0; i < numLayers; i++) {
      feeds[`past_key_values.${i}.key`] = out[`present.${i}.key`]
      feeds[`past_key_values.${i}.value`] = out[`present.${i}.value`]
    }
  }
  return tok.decode(out_ids)
}
