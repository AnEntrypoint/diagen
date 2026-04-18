import fs from 'fs'
import os from 'os'
import path from 'path'
import { getLlama, LlamaChatSession } from 'node-llama-cpp'

const DEFAULT_OLLAMA_BLOB = path.join(
  os.homedir(),
  '.ollama/models/blobs/sha256-74701a8c35f6c8d9a4b91f3f3497643001d63e0c7a84e085bed452548fa88d45'
)
const MODEL_PATH = process.env.LLAMA_MODEL_PATH || DEFAULT_OLLAMA_BLOB
const CONTEXT_SIZE = Number(process.env.LLAMA_CONTEXT_SIZE || 2048)
const GPU_LAYERS = process.env.LLAMA_GPU_LAYERS

let llamaPromise = null
let modelPromise = null

async function getLlamaInstance() {
  if (!llamaPromise) llamaPromise = getLlama()
  return llamaPromise
}

async function getModel() {
  if (modelPromise) return modelPromise
  modelPromise = (async () => {
    if (!fs.existsSync(MODEL_PATH)) throw new Error(`model not found: ${MODEL_PATH}`)
    const llama = await getLlamaInstance()
    const opts = { modelPath: MODEL_PATH }
    if (GPU_LAYERS !== undefined) opts.gpuLayers = Number(GPU_LAYERS)
    console.log(`[llamacpp] loading ${path.basename(MODEL_PATH)} gpu=${llama.gpu || 'none'}`)
    const model = await llama.loadModel(opts)
    console.log(`[llamacpp] loaded`)
    return model
  })()
  return modelPromise
}

async function newSession(system) {
  const model = await getModel()
  const context = await model.createContext({ contextSize: CONTEXT_SIZE })
  return new LlamaChatSession({ contextSequence: context.getSequence(), systemPrompt: system })
}

const GEN_OPTS = { temperature: 0.9, topP: 0.92, maxTokens: 300, repeatPenalty: { penalty: 1.25 }, customStopTriggers: ['\n\n'] }

export async function generate(prompt, system = 'You are a helpful assistant. Be concise.', signal) {
  const t0 = Date.now()
  try {
    const session = await newSession(system)
    const out = await session.prompt(prompt, { ...GEN_OPTS, signal })
    console.log(`[llamacpp] gen ${Date.now()-t0}ms chars=${out.length}`)
    return out
  } catch (err) {
    if (err.name === 'AbortError') { console.log(`[llamacpp] aborted after ${Date.now()-t0}ms`); throw err }
    console.error(`[llamacpp] error after ${Date.now()-t0}ms:`, err.message)
    throw err
  }
}

export async function* generateTokens(prompt, system = 'You are a helpful assistant. Be concise.', signal) {
  const session = await newSession(system)
  const queue = []
  let resolveNext = null
  let done = false
  const onToken = (chunk) => {
    queue.push(chunk)
    if (resolveNext) { const r = resolveNext; resolveNext = null; r() }
  }
  const promptPromise = session.prompt(prompt, {
    ...GEN_OPTS,
    signal,
    onTextChunk: onToken,
  }).then(() => { done = true; if (resolveNext) { const r = resolveNext; resolveNext = null; r() } },
         (e) => { done = true; if (resolveNext) { const r = resolveNext; resolveNext = null; r() } throw e })
  while (true) {
    if (queue.length) { yield queue.shift(); continue }
    if (done) break
    await new Promise(r => { resolveNext = r })
  }
  await promptPromise
}

export async function generateStream(prompt, system = 'You are a helpful assistant. Be concise.') {
  const gen = generateTokens(prompt, system)
  return new ReadableStream({
    async pull(controller) {
      const { value, done } = await gen.next()
      if (done) controller.close()
      else controller.enqueue(new TextEncoder().encode(JSON.stringify({ message: { content: value } }) + '\n'))
    },
  })
}

export async function isAvailable() {
  try {
    if (!fs.existsSync(MODEL_PATH)) return false
    await getModel()
    return true
  } catch {
    return false
  }
}

export default { generate, generateTokens, generateStream, isAvailable }
