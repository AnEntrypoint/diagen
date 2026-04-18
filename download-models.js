import fs from 'fs'
import os from 'os'
import path from 'path'
import { pipeline } from 'stream/promises'
import { createWriteStream } from 'fs'
import { fileURLToPath } from 'url'

const __dirname = path.dirname(fileURLToPath(import.meta.url))

const LLM_SOURCES = [
  {
    name: 'llama3.2-1b.gguf',
    ollamaBlob: path.join(os.homedir(), '.ollama/models/blobs/sha256-74701a8c35f6c8d9a4b91f3f3497643001d63e0c7a84e085bed452548fa88d45'),
  },
]

async function stageLlmModels() {
  const dir = path.join(__dirname, 'models', 'llm')
  fs.mkdirSync(dir, { recursive: true })
  for (const { name, ollamaBlob } of LLM_SOURCES) {
    const dest = path.join(dir, name)
    if (fs.existsSync(dest)) { console.log(`  [${name}] already present`); continue }
    if (fs.existsSync(ollamaBlob)) {
      console.log(`  [${name}] copying from ollama cache (${(fs.statSync(ollamaBlob).size/1e9).toFixed(2)} GB)...`)
      fs.copyFileSync(ollamaBlob, dest)
      console.log(`  [${name}] OK`)
    } else {
      console.log(`  [${name}] missing; set LLAMA_MODEL_PATH or run: ollama pull llama3.2:1b`)
    }
  }
}

const GATEWAYS = [
  'https://dweb.link/ipfs',
  'https://ipfs.io/ipfs',
  'https://w3s.link/ipfs',
  'https://cloudflare-ipfs.com/ipfs',
]

const MODELS = {
  audio2afan: {
    cid: 'bafybeihqgae6fwab5phrimfazd2mvazkpvff5u2ubjkx62n4s5lkwivg5m',
    dir: path.join(__dirname, 'models', 'audio2afan'),
    files: ['config.json', 'model.onnx', 'bs_skin.npz', 'implicit_emo_db.npz', 'model_data.npz', 'solve_data.npz'],
  },
  tts: {
    cid: 'bafybeidyw252ecy4vs46bbmezrtw325gl2ymdltosmzqgx4edjsc3fbofy',
    dir: path.join(__dirname, 'models', 'tts'),
    prefix: 'tts',
    files: ['flow_lm_flow_int8.onnx', 'flow_lm_main_int8.onnx', 'mimi_decoder_int8.onnx', 'mimi_encoder.onnx', 'text_conditioner.onnx', 'tokenizer.model'],
  },
}

async function downloadFile(url, destPath) {
  const res = await fetch(url, { signal: AbortSignal.timeout(300000), redirect: 'follow' })
  if (!res.ok) throw new Error(`HTTP ${res.status}`)
  const ws = createWriteStream(destPath + '.tmp')
  await pipeline(res.body, ws)
  fs.renameSync(destPath + '.tmp', destPath)
  return fs.statSync(destPath).size
}

async function fetchWithFallback(cid, remotePath, destPath) {
  for (const gw of GATEWAYS) {
    const url = `${gw}/${cid}/${remotePath}`
    try {
      const filename = remotePath.split('/').pop()
      process.stdout.write(`  [${filename}] ${gw} ... `)
      const bytes = await downloadFile(url, destPath)
      console.log(`OK (${(bytes / 1e6).toFixed(1)} MB)`)
      return
    } catch (e) {
      console.log(`FAIL (${e.message})`)
    }
  }
  throw new Error(`All gateways failed for ${remotePath}`)
}

async function downloadModel(name, { cid, dir, prefix, files }) {
  fs.mkdirSync(dir, { recursive: true })
  console.log(`\n[${name}] CID: ${cid}`)
  for (const file of files) {
    const dest = path.join(dir, file)
    if (fs.existsSync(dest)) {
      console.log(`  [${file}] already exists, skipping`)
      continue
    }
    const remotePath = prefix ? `${prefix}/${file}` : file
    await fetchWithFallback(cid, remotePath, dest)
  }
}

export async function downloadModels() {
  console.log('Downloading diagen models from IPFS...')
  for (const [name, config] of Object.entries(MODELS)) {
    await downloadModel(name, config)
  }
  console.log('\n[llm] staging GGUF into models/llm/')
  await stageLlmModels()
  console.log('\n[whisper] will auto-download on first use → models/whisper/')
  console.log('[omnivoice] will auto-download on first use → models/omnivoice/')
  console.log('\nAll models ready.')
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  downloadModels().catch(e => { console.error(e.message); process.exit(1) })
}
