import { spawn, execSync } from 'child_process'
import fs from 'fs'

const LOG = 'voice-test.log'

try {
  execSync('taskkill /F /IM node.exe /FI "MEMUSAGE gt 500000" 2>nul', { stdio: 'ignore' })
} catch {}

try { fs.unlinkSync(LOG) } catch {}

const proc = spawn('node', ['server.js'], {
  stdio: ['ignore', 'pipe', 'pipe'],
  env: process.env,
  detached: false,
})

const stream = fs.createWriteStream(LOG, { flags: 'a' })
const filter = /\[(vad|pipe|stream|processor|llamacpp|preamble|discord)\]|error|Error|listening/i

const pipe = (src, tag) => {
  let buf = ''
  src.on('data', (chunk) => {
    buf += chunk.toString()
    const lines = buf.split('\n')
    buf = lines.pop()
    for (const line of lines) {
      stream.write(`[${tag}] ${line}\n`)
      if (filter.test(line)) process.stdout.write(`${line}\n`)
    }
  })
}

pipe(proc.stdout, 'out')
pipe(proc.stderr, 'err')

proc.on('exit', (code) => {
  console.log(`\n[test-voice] server exited code=${code}`)
  stream.end()
  process.exit(code ?? 0)
})

process.on('SIGINT', () => { proc.kill('SIGINT'); setTimeout(() => process.exit(0), 1000) })

console.log('[test-voice] server starting — logs streaming to voice-test.log')
console.log('[test-voice] speak in Discord, relevant events print below')
console.log('[test-voice] Ctrl+C to stop')
