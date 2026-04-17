import fs from 'fs'

const path = process.argv[2] || 'voice-test.log'
const d = fs.readFileSync(path, 'utf8')
const lines = d.split('\n')

const utterances = []
let cur = null

for (const line of lines) {
  const startM = line.match(/\[vad\] 🎤 speech-start uid=(\S+) rms=([\d.]+) botSpeaking=(\S+) preroll=(\d+)ms/)
  if (startM) {
    if (cur) utterances.push(cur)
    cur = { uid: startM[1], startRms: +startM[2], botSpeaking: startM[3] === 'true', preroll: +startM[4], events: [line] }
    continue
  }
  if (!cur) continue
  cur.events.push(line)

  const flushM = line.match(/\[vad\] (flush|⚡ INTERRUPT) uid=\S+ dur=(\d+)ms.*peak=([\d.]+)/)
  if (flushM) { cur.flushKind = flushM[1]; cur.dur = +flushM[2]; cur.peak = +flushM[3] }

  const rejectM = line.match(/\[vad\] ✗ reject uid=\S+ dur=(\d+)ms peak=([\d.]+)/)
  if (rejectM) { cur.flushKind = 'reject'; cur.dur = +rejectM[1]; cur.peak = +rejectM[2] }

  const textM = line.match(/\[pipe\] uid=\S+ \([^)]+\) text="([^"]*)" conf=([\d.]+)/)
  if (textM) { cur.text = textM[1]; cur.conf = +textM[2] }

  const ftM = line.match(/first-token (\d+)ms/)
  if (ftM) cur.firstTokenMs = +ftM[1]

  const faM = line.match(/first-audio (\d+)ms/)
  if (faM) cur.firstAudioMs = +faM[1]

  const respM = line.match(/stream-gen total (\d+)ms → "([^"]*)"/)
  if (respM) { cur.genMs = +respM[1]; cur.response = respM[2] }

  const doneM = line.match(/◀ done uid=\S+ totalMs=(\d+)/)
  if (doneM) { cur.totalMs = +doneM[1]; utterances.push(cur); cur = null }
}
if (cur) utterances.push(cur)

console.log(`=== ${utterances.length} utterances ===\n`)
for (const [i, u] of utterances.entries()) {
  const head = `#${i+1} [${u.flushKind || 'incomplete'}] uid=${u.uid} dur=${u.dur ?? '?'}ms peak=${u.peak?.toFixed(3) ?? '?'} preroll=${u.preroll}ms`
  console.log(head)
  if (u.text !== undefined) console.log(`  heard: "${u.text}" (conf=${u.conf})`)
  if (u.response) console.log(`  said:  "${u.response}"`)
  const timing = []
  if (u.firstTokenMs) timing.push(`ft=${u.firstTokenMs}`)
  if (u.firstAudioMs) timing.push(`fa=${u.firstAudioMs}`)
  if (u.genMs) timing.push(`gen=${u.genMs}`)
  if (u.totalMs) timing.push(`total=${u.totalMs}`)
  if (timing.length) console.log(`  timing: ${timing.join(' ')}ms`)
  console.log()
}

const heard = utterances.filter(u => u.text && u.text !== '[no speech detected]' && u.text.length >= 2)
const replied = utterances.filter(u => u.response)
const rejected = utterances.filter(u => u.flushKind === 'reject')
console.log(`summary: ${utterances.length} utterances | ${heard.length} transcribed | ${replied.length} replied | ${rejected.length} rejected`)
if (replied.length) {
  const avgFa = replied.filter(u => u.firstAudioMs).reduce((s,u) => s + u.firstAudioMs, 0) / replied.filter(u => u.firstAudioMs).length
  console.log(`avg first-audio: ${avgFa.toFixed(0)}ms`)
}
