import express from 'express'
import fs from 'fs'
import path from 'path'
import { fileURLToPath } from 'url'
import { createRequire } from 'module'
import { Audio2FaceCore } from './audio2afan_core.mjs'
import ort from 'onnxruntime-node'
import { ARKIT_NAMES, encodeWAV, resampleAudio, buildAfan } from './server-utils.mjs'
import { synthesize as synthesizeOmniVoice } from './pocket-tts-bridge.js'
import { generate as generateLLM, isAvailable as isLLMAvailable } from './llm-llamacpp.js'
import os from 'os'

// Load environment variables from .env
const __dirname = path.dirname(fileURLToPath(import.meta.url))
const envPath = path.join(__dirname, '.env')
if (fs.existsSync(envPath)) {
  const envContent = fs.readFileSync(envPath, 'utf8')
  envContent.split('\n').forEach(line => {
    const [key, ...valueParts] = line.split('=')
    if (key && !key.startsWith('#') && valueParts.length > 0) {
      const value = valueParts.join('=').trim()
      if (!process.env[key.trim()]) {
        process.env[key.trim()] = value
      }
    }
  })
}

let initDiscordBot = null
let sendMessage = null
let connectToVoiceChannel = null
let disconnectFromVoiceChannel = null
let getDebugState = null
let setVoiceEmbedding = null
let getDiscordClient = null

const require = createRequire(import.meta.url)
const ORT_CPUS = os.cpus().length
const origOrtCreate = ort.InferenceSession.create.bind(ort.InferenceSession)
ort.InferenceSession.create = async function(modelPath, options = {}) {
  const explicit = options.executionProviders
  const providers = explicit || ['cpu']
  const wantDml = !explicit || (!providers.includes('cpu') || providers.length > 1)
  const patched = {
    ...options,
    executionProviders: (wantDml && !providers.includes('dml')) ? ['dml', ...providers] : providers,
    intraOpNumThreads: Math.max(options.intraOpNumThreads || 0, ORT_CPUS),
    interOpNumThreads: Math.max(options.interOpNumThreads || 0, Math.floor(ORT_CPUS / 4)),
    executionMode: 'parallel',
    enableCpuMemArena: true,
    enableMemPattern: true,
    graphOptimizationLevel: options.graphOptimizationLevel || 'all',
  }
  return origOrtCreate(modelPath, patched)
}

const app = express()
const port = process.env.PORT || 8080
app.use(express.json({ limit: '50mb' }))
app.use((req, res, next) => {
  res.setHeader('Cross-Origin-Opener-Policy', 'same-origin')
  res.setHeader('Cross-Origin-Embedder-Policy', 'require-corp')
  res.setHeader('Access-Control-Allow-Origin', '*')
  res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type')
  if (req.method === 'OPTIONS') return res.sendStatus(200)
  next()
})
const CLEETUS_WAV = path.join(__dirname, 'voices', 'cleetus.wav')
const TTS_MODELS_DIR = path.join(__dirname, 'models', 'tts')
app.get('/', (req, res) => res.sendFile(path.join(__dirname, 'index.html')))
app.get('/client.js', (req, res) => res.sendFile(path.join(__dirname, 'client.js')))
app.get('/animation-core.mjs', (req, res) => res.sendFile(path.join(__dirname, 'animation-core.mjs')))
app.get('/idle-animator.mjs', (req, res) => res.sendFile(path.join(__dirname, 'idle-animator.mjs')))
app.get('/facial-player.mjs', (req, res) => res.sendFile(path.join(__dirname, 'facial-player.mjs')))
app.get('/llm-worker.js', (req, res) => res.sendFile(path.join(__dirname, 'llm-worker.js')))
app.get('/Cleetus.vrm', (req, res) => res.sendFile(path.join(__dirname, 'Cleetus.vrm')))
app.use('/node_modules', express.static(path.join(__dirname, 'node_modules')))
const DEMO_DIR = path.join(__dirname, 'gh-pages-src', 'demo')
const VOICES_DIR = path.join(__dirname, 'voices')
app.get('/demo/voices/manifest.json', (req, res) => {
  const files = fs.existsSync(VOICES_DIR) ? fs.readdirSync(VOICES_DIR).filter(f => f.endsWith('.wav')) : []
  res.json(files)
})
app.use('/demo/voices', express.static(VOICES_DIR))
app.use('/demo', express.static(DEMO_DIR))
const SAMPLE_RATE = 24000, A2F_SAMPLE_RATE = 16000
let a2f = null, voiceEmbedding = null

async function loadA2F() {
  if (a2f) return a2f
  const audio2afanDir = path.join(__dirname, 'models', 'audio2afan')
  a2f = new Audio2FaceCore({ ort })
  await a2f.loadConfigFile(path.join(audio2afanDir, 'config.json'))
  await a2f.loadModel(path.join(audio2afanDir, 'model.onnx'))
  await a2f.loadSolveData(audio2afanDir)
  console.log('[a2f] Loaded')
  return a2f
}
async function loadVoiceEmbedding() {
  if (voiceEmbedding) return voiceEmbedding
  if (!fs.existsSync(CLEETUS_WAV)) {
    console.warn('[voice] No voice file found at', CLEETUS_WAV)
    return null
  }
  console.log('[voice] Voice reference loaded:', CLEETUS_WAV)
  voiceEmbedding = CLEETUS_WAV
  return voiceEmbedding
}
app.post('/api/generate', async (req, res) => {
  try {
    const { text } = req.body
    if (!text) return res.status(400).json({ error: 'text required' })
    await loadA2F()
    const voiceEmb = await loadVoiceEmbedding()
    if (!voiceEmb) return res.status(500).json({ error: 'Voice embedding not available' })
    const startTime = performance.now()

    const audioFloat = await synthesizeOmniVoice(text, voiceEmb, 'reference speech')
    const [audioWav, audio16k] = await Promise.all([
      Promise.resolve(encodeWAV(audioFloat, SAMPLE_RATE)),
      Promise.resolve(resampleAudio(audioFloat, SAMPLE_RATE, A2F_SAMPLE_RATE))
    ])
    const fps = 30
    const stride = Math.round(A2F_SAMPLE_RATE / fps)
    const bufferLen = a2f.bufferLen
    const bufferOfs = a2f.bufferOfs
    const predictionDelay = a2f.faceParams?.prediction_delay || 0

    const results = []
    let lastBs = null
    for (let offset = 0; offset + bufferLen <= audio16k.length; offset += stride) {
      const chunk = audio16k.slice(offset, offset + bufferLen)
      const result = await a2f.runInference(chunk)
      if (result.blendshapes && lastBs) {
        result.blendshapes = a2f.smoothBlendshapes(lastBs, result.blendshapes)
      }
      lastBs = result.blendshapes
      results.push({
        time: (offset + bufferOfs) / A2F_SAMPLE_RATE + predictionDelay,
        blendshapes: result.blendshapes
      })
    }
    const audioDuration = audioFloat.length / SAMPLE_RATE
    const targetNumFrames = Math.ceil(audioDuration * fps)
    const frames = new Array(targetNumFrames)
    let cursor = 0
    for (let frameIdx = 0; frameIdx < targetNumFrames; frameIdx++) {
      const frameTime = frameIdx / fps
      const frame = new Float32Array(52)
      if (results.length === 0 || frameTime < results[0].time) { frames[frameIdx] = frame; continue }
      while (cursor < results.length - 1 && results[cursor + 1].time <= frameTime) cursor++

      const curr = results[cursor]
      const next = results[Math.min(cursor + 1, results.length - 1)]
      const dt = next.time - curr.time
      const t = dt > 0 ? Math.max(0, Math.min(1, (frameTime - curr.time) / dt)) : 0

      for (let k = 0; k < 52; k++) {
        frame[k] = curr.blendshapes[k].value + (next.blendshapes[k].value - curr.blendshapes[k].value) * t
      }
      frames[frameIdx] = frame
    }

    const animBuffer = buildAfan(frames, fps)
    const duration = audioFloat.length / SAMPLE_RATE
    const genTime = ((performance.now() - startTime) / 1000).toFixed(1)
    const rtfx = (duration / parseFloat(genTime)).toFixed(1)

    console.log(`[generate] "${text.slice(0, 30)}..." - ${duration.toFixed(1)}s in ${genTime}s (${rtfx}x realtime)`)
    res.json({
      audio: audioWav.toString('base64'),
      animation: animBuffer.toString('base64'),
      duration,
    })
  } catch (err) {
    console.error('Generate error:', err)
    res.status(500).json({ error: err.message })
  }
})
async function ensureModels() {
  const { downloadModels } = await import('./download-models.js')
  await downloadModels()
}

app.post('/api/chat', async (req, res) => {
  try {
    const { prompt, system } = req.body
    if (!prompt) return res.status(400).json({ error: 'prompt required' })
    const available = await isLLMAvailable()
    if (!available) return res.status(503).json({ error: 'LLM not available (llama.cpp model failed to load)' })
    const response = await generateLLM(prompt, system)
    res.json({ response })
  } catch (err) {
    console.error('[llm] error:', err)
    res.status(500).json({ error: err.message })
  }
})

let _setCharacterCard = null
app.post('/api/character/card', (req, res) => {
  try {
    const card = req.body
    if (!card || typeof card !== 'object') return res.status(400).json({ error: 'card JSON required' })
    if (_setCharacterCard) {
      _setCharacterCard(card)
      res.json({ success: true })
    } else {
      res.status(503).json({ error: 'Discord processor not loaded' })
    }
  } catch (err) {
    res.status(500).json({ error: err.message })
  }
})

// Discord API endpoints
app.post('/api/discord/voice/connect', async (req, res) => {
  if (!connectToVoiceChannel) return res.status(503).json({ error: 'Discord not enabled' })
  try {
    const { guildId, channelId } = req.body
    if (!guildId || !channelId) {
      return res.status(400).json({ error: 'guildId and channelId required' })
    }
    console.log('[api] Voice connect request for guild', guildId, 'channel', channelId)
    const connectPromise = connectToVoiceChannel(guildId, channelId)
    const timeoutPromise = new Promise((_, reject) => setTimeout(() => reject(new Error('Connection timeout after 90s')), 90000))
    await Promise.race([connectPromise, timeoutPromise])
    res.json({ success: true })
  } catch (err) {
    console.error('[api] Discord voice connect error:', err)
    res.status(500).json({ error: err.message })
  }
})

app.post('/api/discord/voice/disconnect', (req, res) => {
  if (!disconnectFromVoiceChannel) return res.status(503).json({ error: 'Discord not enabled' })
  try {
    disconnectFromVoiceChannel()
    res.json({ success: true })
  } catch (err) {
    console.error('[api] Discord voice disconnect error:', err)
    res.status(500).json({ error: err.message })
  }
})

app.post('/api/discord/message', async (req, res) => {
  if (!sendMessage) return res.status(503).json({ error: 'Discord not enabled' })
  try {
    const { channelId, message } = req.body
    if (!channelId || !message) {
      return res.status(400).json({ error: 'channelId and message required' })
    }
    await sendMessage(channelId, message)
    res.json({ success: true })
  } catch (err) {
    console.error('[api] Discord message error:', err)
    res.status(500).json({ error: err.message })
  }
})

app.get('/debug/discord', (req, res) => {
  if (!getDebugState) return res.status(503).json({ error: 'Discord not enabled' })
  try {
    const state = getDebugState()
    res.json(state)
  } catch (err) {
    console.error('[api] Debug discord error:', err)
    res.status(500).json({ error: err.message })
  }
})

app.get('/debug/guild/:guildId/channel/:channelId', async (req, res) => {
  try {
    const { guildId, channelId } = req.params
    if (!getDiscordClient) return res.status(503).json({ error: 'Discord not initialized' })

    const client = getDiscordClient()
    if (!client) return res.status(503).json({ error: 'Discord client not available' })

    const guild = await client.guilds.fetch(guildId)
    const channel = await guild.channels.fetch(channelId)
    const botMember = await guild.members.fetchMe()

    const voicePermissions = channel.permissionsFor(botMember)

    res.json({
      guild: { id: guild.id, name: guild.name },
      channel: { id: channel.id, name: channel.name, type: channel.type },
      botMember: { id: botMember.id, nickname: botMember.nickname, roles: botMember.roles.cache.map(r => r.name) },
      voicePermissions: {
        connect: voicePermissions.has('Connect'),
        speak: voicePermissions.has('Speak'),
        useVoiceActivity: voicePermissions.has('UseVoiceActivity'),
        all: voicePermissions.toArray()
      }
    })
  } catch (err) {
    console.error('[api] Guild/channel debug error:', err)
    res.status(500).json({ error: err.message })
  }
})

async function start() {
  const discordOnly = (process.env.DISCORD_TOKEN || process.env.DISCORD_BOT_TOKEN) && process.env.DEMO !== '1'
  if (!discordOnly) {
    await ensureModels()
    await loadA2F()
  }
  await loadVoiceEmbedding()

  // Warm up TTS to avoid first-call timeout (model download ~5-10min)
  if (process.env.WARMUP_TTS !== 'false' && !discordOnly) {
    try {
      console.log('[server] Warming up OmniVoice TTS (first-time model download)...')
      const warmupStart = performance.now()
      await synthesizeOmniVoice('Server starting', CLEETUS_WAV, 'test warmup')
      const warmupTime = ((performance.now() - warmupStart) / 1000).toFixed(1)
      console.log(`[server] TTS warmup complete (${warmupTime}s) - subsequent calls will be fast`)
    } catch (err) {
      console.warn('[server] TTS warmup failed (non-critical):', err.message)
    }
  }

  // Lazy load Discord modules only if DISCORD_TOKEN is set
  if (process.env.DISCORD_TOKEN || process.env.DISCORD_BOT_TOKEN) {
    try {
      const discordHandler = await import('./discord-handler.js')
      const discordProcessor = await import('./discord-voice-processor.js')

      initDiscordBot = discordHandler.initDiscordBot
      sendMessage = discordHandler.sendMessage
      connectToVoiceChannel = discordHandler.connectToVoiceChannel
      disconnectFromVoiceChannel = discordHandler.disconnectFromVoiceChannel
      getDebugState = discordHandler.getDebugState
      getDiscordClient = discordHandler.getDiscordClient
      setVoiceEmbedding = discordProcessor.setVoiceEmbedding
      const setCharacterCard = discordProcessor.setCharacterCard
      _setCharacterCard = setCharacterCard
      const getCharacterSystemPrompt = discordProcessor.getCharacterSystemPrompt

      setVoiceEmbedding(CLEETUS_WAV)
      console.log('[server] Voice reference path set for Discord processor')

      const preambleCache = await import('./preamble-cache.js')
      const refTextPath = CLEETUS_WAV.replace(/\.wav$/i, '.txt')
      const refText = fs.existsSync(refTextPath) ? fs.readFileSync(refTextPath, 'utf8').trim() : null
      preambleCache.warmup(CLEETUS_WAV, refText).catch(err => console.warn('[server] preamble warmup:', err.message))

      const cleetusCard = CLEETUS_WAV.replace(/\.wav$/i, '.json')
      if (fs.existsSync(cleetusCard)) {
        try {
          setCharacterCard(JSON.parse(fs.readFileSync(cleetusCard, 'utf8')))
          console.log('[server] Loaded default character card from', cleetusCard)
        } catch (err) {
          console.warn('[server] Failed to load character card:', err.message)
        }
      }

      try {
        const llm = await import('./llm-llamacpp.js')
        llm.warmup(getCharacterSystemPrompt() || undefined).catch(err => console.warn('[server] llm warmup:', err.message))
      } catch (err) {
        console.warn('[server] llm warmup import failed:', err.message)
      }

      // Initialize Discord bot
      const onCommand = async (userId, prompt) => {
        const available = await isLLMAvailable()
        if (!available) return `[LLM offline] Received: ${prompt}`
        return generateLLM(prompt, getCharacterSystemPrompt() || undefined)
      }
      const onUserAudio = (userId, pcmChunk) => {}
      const autoGuild = process.env.GUILD_ID
      const autoChannel = process.env.CHANNEL_ID
      const onBotReady = autoGuild && autoChannel ? async () => {
        try {
          await connectToVoiceChannel(autoGuild, autoChannel)
          console.log(`[server] ✓ Auto-joined voice channel ${autoChannel}`)
        } catch (err) {
          console.error('[server] ⚠ Auto-join failed:', err.message)
          console.log('[server] To manually connect:')
          console.log(`[server]   curl -X POST http://localhost:8080/api/discord/voice/connect -H "Content-Type: application/json" -d '{"guildId":"${autoGuild}","channelId":"${autoChannel}"}'`)
        }
      } : null
      await initDiscordBot(onUserAudio, onCommand, onBotReady)
    } catch (err) {
      console.error('[server] Failed to load Discord modules:', err.message)
    }
  }

  app.listen(port, '0.0.0.0', () => {
    console.log(`diagen server running on http://localhost:${port}`)
  })
}
start().catch(err => {
  console.error('Startup failed:', err)
  process.exit(1)
})
