import fs from 'fs'
import path from 'path'
import { fileURLToPath } from 'url'

const __dirname = path.dirname(fileURLToPath(import.meta.url))

const REQUIRED = {
  tts: {
    dir: path.join(__dirname, 'models', 'tts'),
    files: ['flow_lm_flow_int8.onnx', 'flow_lm_main_int8.onnx', 'mimi_decoder_int8.onnx', 'mimi_encoder.onnx', 'text_conditioner.onnx', 'tokenizer.model'],
  },
  llm: {
    dir: path.join(__dirname, 'models', 'llm'),
    pattern: /\.gguf$/,
    hint: 'Run: git lfs pull',
  },
}

function isLfsPointer(file) {
  try {
    const buf = Buffer.alloc(64)
    const fd = fs.openSync(file, 'r')
    const n = fs.readSync(fd, buf, 0, 64, 0)
    fs.closeSync(fd)
    return buf.slice(0, n).toString('utf8').startsWith('version https://git-lfs.github.com')
  } catch {
    return false
  }
}

export async function downloadModels() {
  console.log('Verifying models under models/ ...')
  let missing = 0
  for (const [name, cfg] of Object.entries(REQUIRED)) {
    if (!fs.existsSync(cfg.dir)) { console.log(`[${name}] ✗ dir missing: ${cfg.dir}`); missing++; continue }
    if (cfg.pattern) {
      const matches = fs.readdirSync(cfg.dir).filter(f => cfg.pattern.test(f))
      if (!matches.length) { console.log(`[${name}] ✗ no files matching ${cfg.pattern}. ${cfg.hint || ''}`); missing++; continue }
      for (const f of matches) {
        const p = path.join(cfg.dir, f)
        if (isLfsPointer(p)) { console.log(`[${name}] ✗ ${f} is LFS pointer; run: git lfs pull`); missing++ }
        else console.log(`[${name}] ✓ ${f} (${(fs.statSync(p).size/1e6).toFixed(1)} MB)`)
      }
    } else {
      for (const f of cfg.files) {
        const p = path.join(cfg.dir, f)
        if (!fs.existsSync(p)) { console.log(`[${name}] ✗ missing ${f}`); missing++ }
        else if (isLfsPointer(p)) { console.log(`[${name}] ✗ ${f} is LFS pointer; run: git lfs pull`); missing++ }
        else console.log(`[${name}] ✓ ${f} (${(fs.statSync(p).size/1e6).toFixed(1)} MB)`)
      }
    }
  }
  if (missing) { console.log(`\n${missing} file(s) missing. If this is a fresh clone, run: git lfs pull`); process.exit(1) }
  console.log('\nAll models present.')
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  downloadModels().catch(e => { console.error(e.message); process.exit(1) })
}
