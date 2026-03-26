import fs from 'fs'
import path from 'path'
import { createRequire } from 'module'
import ort from 'onnxruntime-node'
import { buildTokenizer } from './tokenizer.mjs'

const require = createRequire(import.meta.url)
const { qwenDir } = require('sttttsmodels')

const MODEL_DIR = qwenDir
const EOS_IDS = new Set([248046, 248044])

let embedSess = null, decoderSess = null, tok = null, loadP = null
let FULL_ATTN = null, NUM_LAYERS = null, NUM_KV_HEADS = null, HEAD_DIM = null

export async function loadQwenModel() {
  if (embedSess) return
  if (loadP) return loadP
  loadP = (async () => {
    const cfg = JSON.parse(fs.readFileSync(path.join(MODEL_DIR, 'config.json'), 'utf8'))
    const tc = cfg.text_config
    NUM_LAYERS = tc.num_hidden_layers
    NUM_KV_HEADS = tc.num_key_value_heads
    HEAD_DIM = tc.head_dim
    const layerTypes = tc.layer_types
    FULL_ATTN = new Set(layerTypes.map((t, i) => t === 'full_attention' ? i : -1).filter(i => i >= 0))
    tok = buildTokenizer(JSON.parse(fs.readFileSync(path.join(MODEL_DIR, 'tokenizer.json'), 'utf8')))
    const opts = { executionProviders: ['cpu'], graphOptimizationLevel: 'all' }
    console.log('[qwen] loading embed_tokens...')
    embedSess = await ort.InferenceSession.create(path.join(MODEL_DIR, 'onnx/embed_tokens_q4.onnx'), opts)
    console.log('[qwen] loading decoder...')
    decoderSess = await ort.InferenceSession.create(path.join(MODEL_DIR, 'onnx/decoder_model_merged_q4.onnx'), opts)
    console.log('[qwen] ready')
  })()
  return loadP
}

function initState() {
  const s = {}
  for (let i = 0; i < NUM_LAYERS; i++) {
    if (FULL_ATTN.has(i)) {
      s[`past_key_values.${i}.key`] = new ort.Tensor('float32', new Float32Array(0), [1, NUM_KV_HEADS, 0, HEAD_DIM])
      s[`past_key_values.${i}.value`] = new ort.Tensor('float32', new Float32Array(0), [1, NUM_KV_HEADS, 0, HEAD_DIM])
    } else {
      s[`past_conv.${i}`] = new ort.Tensor('float32', new Float32Array(1 * 6144 * 4), [1, 6144, 4])
      s[`past_recurrent.${i}`] = new ort.Tensor('float32', new Float32Array(1 * 16 * 128 * 128), [1, 16, 128, 128])
    }
  }
  return s
}

function syncState(state, out) {
  for (let i = 0; i < NUM_LAYERS; i++) {
    if (FULL_ATTN.has(i)) {
      state[`past_key_values.${i}.key`] = out[`present.${i}.key`]
      state[`past_key_values.${i}.value`] = out[`present.${i}.value`]
    } else {
      state[`past_conv.${i}`] = out[`present_conv.${i}`]
      state[`past_recurrent.${i}`] = out[`present_recurrent.${i}`]
    }
  }
}

export async function generateDialog(prompt, { maxNewTokens = 200, system = 'You are a helpful dialog assistant. Be concise.' } = {}) {
  await loadQwenModel()
  const ids = tok.tokenize(`<|im_start|>system\n${system}<|im_end|>\n<|im_start|>user\n${prompt}<|im_end|>\n<|im_start|>assistant\n`)
  const state = initState()
  let seqLen = ids.length

  const { inputs_embeds } = await embedSess.run({
    input_ids: new ort.Tensor('int64', BigInt64Array.from(ids.map(BigInt)), [1, seqLen])
  })
  const posSeq = BigInt64Array.from(ids.map((_, i) => BigInt(i)))
  let feeds = {
    inputs_embeds,
    attention_mask: new ort.Tensor('int64', BigInt64Array.from(ids.map(() => 1n)), [1, seqLen]),
    position_ids: new ort.Tensor('int64', [...posSeq, ...posSeq, ...posSeq], [3, 1, seqLen]),
    ...state,
  }

  const out_ids = []
  for (let step = 0; step < maxNewTokens; step++) {
    const out = await decoderSess.run(feeds)
    const logits = out.logits.data
    const vSize = out.logits.dims[2]
    const offset = (out.logits.dims[1] - 1) * vSize
    let maxV = -Infinity, nextId = 0
    for (let v = 0; v < vSize; v++) { if (logits[offset + v] > maxV) { maxV = logits[offset + v]; nextId = v } }
    if (EOS_IDS.has(nextId)) break
    out_ids.push(nextId)
    syncState(state, out)
    seqLen++
    const pos = BigInt(seqLen - 1)
    const { inputs_embeds: nextEmbed } = await embedSess.run({
      input_ids: new ort.Tensor('int64', BigInt64Array.from([BigInt(nextId)]), [1, 1])
    })
    feeds = {
      inputs_embeds: nextEmbed,
      attention_mask: new ort.Tensor('int64', BigInt64Array.from(new Array(seqLen).fill(1n)), [1, seqLen]),
      position_ids: new ort.Tensor('int64', [pos, pos, pos], [3, 1, 1]),
      ...state,
    }
  }
  return tok.decode(out_ids)
}
