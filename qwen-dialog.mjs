import { createRequire } from 'module'
import { Qwen3_5ForCausalLM, AutoTokenizer, env } from '@huggingface/transformers'

const require = createRequire(import.meta.url)
const { modelsDir } = require('sttttsmodels')

const MODEL_ID = 'onnx-community/Qwen3.5-0.8B-ONNX'

env.localModelPath = modelsDir + '/qwen/'
env.allowRemoteModels = false
env.allowLocalModels = true

let model = null, tokenizer = null, loadP = null

export async function loadQwenModel() {
  if (model) return
  if (loadP) return loadP
  loadP = (async () => {
    console.log('[qwen] loading tokenizer...')
    tokenizer = await AutoTokenizer.from_pretrained(MODEL_ID, { local_files_only: true })
    console.log('[qwen] loading ONNX model...')
    model = await Qwen3_5ForCausalLM.from_pretrained(MODEL_ID, {
      local_files_only: true,
      dtype: { embed_tokens: 'q4f16', decoder_model_merged: 'q4f16' },
      device: 'cpu',
    })
    console.log('[qwen] ready')
  })()
  return loadP
}

export async function generateDialog(prompt, { maxNewTokens = 200, system = 'You are a helpful dialog assistant. Be concise.' } = {}) {
  await loadQwenModel()
  const messages = [
    { role: 'system', content: system },
    { role: 'user', content: prompt },
  ]
  const text = tokenizer.apply_chat_template(messages, { tokenize: false, add_generation_prompt: true, enable_thinking: false })
  const inputs = tokenizer(text, { return_tensors: 'pt' })
  const inputLen = inputs.input_ids.dims[1]
  const output = await model.generate({ ...inputs, max_new_tokens: maxNewTokens })
  const newTokens = output.slice(null, [inputLen, null])
  const [decoded] = tokenizer.batch_decode(newTokens, { skip_special_tokens: true })
  return decoded
}
