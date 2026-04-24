import { createTTSEngine, splitTextIntoChunks } from 'https://esm.sh/streamtts@latest'

const MODEL_BASE_PATH = 'https://raw.githubusercontent.com/AnEntrypoint/streamtts/models/'
const SAMPLE_RATE = 24000

const engine = createTTSEngine(self)
let activeSpeakerId = null
let aborted = false

async function loadVoice(voiceName) {
  self.postMessage({ type: 'status', status: `Encoding speaker: ${voiceName}` })
  const resp = await fetch(`./voices/${voiceName}.wav`)
  if (!resp.ok) throw new Error(`voice fetch failed: ${voiceName}.wav`)
  const audioCtx = new OfflineAudioContext(1, 1, SAMPLE_RATE)
  const decoded = await audioCtx.decodeAudioData(await resp.arrayBuffer())
  const mono = decoded.getChannelData(0)
  await engine.encodeSpeaker(voiceName, mono)
  activeSpeakerId = voiceName
  self.postMessage({ type: 'loaded' })
}

async function generate(text) {
  if (!activeSpeakerId) throw new Error('No speaker encoded — load a voice first')
  aborted = false
  for (const chunk of splitTextIntoChunks(text)) {
    if (aborted) break
    const waveform = await engine.generate(chunk.text, activeSpeakerId)
    if (aborted) break
    const buf = waveform.buffer.slice(waveform.byteOffset, waveform.byteOffset + waveform.byteLength)
    self.postMessage({ type: 'audio_chunk', data: buf }, [buf])
  }
  if (!aborted) self.postMessage({ type: 'stream_ended' })
}

self.onmessage = async (e) => {
  const { type, data } = e.data
  try {
    if (type === 'load') {
      self.postMessage({ type: 'status', status: 'Loading Chatterbox Turbo model…' })
      engine.configure({ modelBasePath: MODEL_BASE_PATH, allowRemoteModels: false })
      const info = await engine.load({
        onProgress: (p) => {
          if (p.status === 'progress') {
            const pct = p.progress != null ? ` ${Math.round(p.progress)}%` : ''
            self.postMessage({ type: 'status', status: `Loading: ${p.file}${pct}` })
          }
        },
      })
      self.postMessage({ type: 'status', status: `Model loaded (${info.device})` })
      const manifest = await fetch('./voices/manifest.json').then((r) => r.json())
      const voices = manifest.map((f) => f.replace('.wav', ''))
      self.postMessage({ type: 'voices_loaded', voices, defaultVoice: voices[0] || 'cleetus' })
    } else if (type === 'load_voice') {
      await loadVoice(data?.voice ?? e.data.voice)
    } else if (type === 'generate') {
      await generate(data?.text ?? e.data.data?.text)
    } else if (type === 'cancel') {
      aborted = true
    }
  } catch (err) {
    self.postMessage({ type: 'error', error: err.message })
  }
}
