import { ChannelType } from 'discord.js'
import { createClient, joinDiscordVoice, subscribeToSpeaker, leaveVoice } from './discord-bot-client.js'
import { processUserAudio } from './discord-voice-processor.js'
import { initVoicePlayer, pushAudioFrame } from './discord-voice-player.js'

let discordClient = null
let isConnected = false
let currentChannelState = { guildId: null, channelId: null }
let lastError = null
let messageCount = 0
let processingQueue = []
let _onUserAudio = null
let _onCommand = null

const SILENCE_THRESHOLD = 0.01
const SILENCE_DURATION_MS = 1500
const MIN_UTTERANCE_MS = 500
const MAX_UTTERANCE_MS = 30000
const SAMPLE_RATE = 48000

const userBuffers = new Map()

function getOrCreateBuffer(userId) {
  if (!userBuffers.has(userId)) {
    userBuffers.set(userId, { chunks: [], startTime: 0, lastVoiceTime: 0, processing: false })
  }
  return userBuffers.get(userId)
}

function rms(f32) {
  let sum = 0
  for (let i = 0; i < f32.length; i++) sum += f32[i] * f32[i]
  return Math.sqrt(sum / f32.length)
}

async function handleUtterance(userId, chunks) {
  const totalLen = chunks.reduce((s, c) => s + c.length, 0)
  const merged = new Float32Array(totalLen)
  let offset = 0
  for (const chunk of chunks) {
    merged.set(chunk, offset)
    offset += chunk.length
  }
  const int16 = new Int16Array(merged.length)
  for (let i = 0; i < merged.length; i++) {
    const v = Math.max(-1, Math.min(1, merged[i]))
    int16[i] = v < 0 ? v * 0x8000 : v * 0x7FFF
  }
  const pcmBuffer = Buffer.from(int16.buffer)
  const durationMs = (totalLen / SAMPLE_RATE) * 1000
  console.log(`[voice] userId=${userId} utterance: ${(durationMs / 1000).toFixed(1)}s, ${totalLen} samples`)

  processingQueue.push({ userId, startTime: Date.now(), samples: totalLen })
  try {
    const audioOutput = await processUserAudio(pcmBuffer, SAMPLE_RATE, userId)
    pushAudioFrame(audioOutput)
    console.log(`[voice] userId=${userId} response sent: ${audioOutput.length} bytes`)
  } catch (err) {
    console.error(`[voice] userId=${userId} pipeline error: ${err.message}`)
    lastError = { message: err.message, timestamp: Date.now(), userId }
  } finally {
    processingQueue = processingQueue.filter(p => p.userId !== userId || p.startTime !== processingQueue.find(q => q.userId === userId)?.startTime)
  }
}

function onPcmChunk(userId, f32) {
  const buf = getOrCreateBuffer(userId)
  if (buf.processing) return

  const now = Date.now()
  const level = rms(f32)
  const isSpeech = level > SILENCE_THRESHOLD

  if (isSpeech) {
    if (buf.chunks.length === 0) buf.startTime = now
    buf.lastVoiceTime = now
    buf.chunks.push(new Float32Array(f32))
  }

  if (buf.chunks.length === 0) return

  const utteranceDuration = now - buf.startTime
  const silenceDuration = now - buf.lastVoiceTime
  const shouldFlush = silenceDuration >= SILENCE_DURATION_MS || utteranceDuration >= MAX_UTTERANCE_MS

  if (!shouldFlush) return

  if (utteranceDuration < MIN_UTTERANCE_MS) {
    buf.chunks = []
    return
  }

  const chunks = buf.chunks
  buf.chunks = []
  buf.processing = true
  handleUtterance(userId, chunks).finally(() => { buf.processing = false })
}

function subscribeUser(userId) {
  subscribeToSpeaker(userId, onPcmChunk)
  console.log(`[voice] subscribed to user ${userId}`)
}

async function initDiscordBot(onUserAudio, onCommand, onReady) {
  const token = process.env.DISCORD_TOKEN || process.env.DISCORD_BOT_TOKEN
  if (!token) {
    console.log('[discord] DISCORD_TOKEN not set, Discord bot disabled')
    return
  }

  _onUserAudio = onUserAudio
  _onCommand = onCommand
  discordClient = createClient()

  discordClient.on('ready', () => {
    console.log('[discord] ✓ Bot ready - logged in as', discordClient.user.tag, `(ID: ${discordClient.user.id})`)
    isConnected = true
    if (onReady) onReady()
  })

  discordClient.on('error', (err) => {
    console.error('[discord] Client error:', err.message)
    lastError = { message: err.message, timestamp: Date.now() }
  })

  discordClient.on('warn', (warn) => {
    console.warn('[discord] Client warning:', warn)
  })

  discordClient.on('messageCreate', async (message) => {
    if (message.author.bot || !message.guild) return

    if (message.content.startsWith('!join ')) {
      const channelId = message.content.slice(6).trim()
      if (!channelId) { await message.reply('Usage: !join <channel-id>'); return }
      try {
        await handleJoinCommand(message.guildId, channelId)
        await message.reply(`Joining voice channel ${channelId}...`)
      } catch (err) {
        await message.reply('Error joining channel: ' + err.message)
      }
    }

    if (message.content.startsWith('!diagen ')) {
      const prompt = message.content.slice(8).trim()
      if (!prompt) { await message.reply('Usage: !diagen <prompt>'); return }
      try {
        await message.channel.sendTyping()
        messageCount++
        const response = await _onCommand(message.author.id, prompt)
        const chunks = []
        for (let i = 0; i < response.length; i += 2000) chunks.push(response.slice(i, i + 2000))
        for (const chunk of chunks) await message.reply(chunk)
      } catch (err) {
        await message.reply('Error: ' + err.message)
      }
    }
  })

  discordClient.on('voiceStateUpdate', (oldState, newState) => {
    if (!newState.member?.user?.bot && newState.channelId === currentChannelState.channelId && newState.guild?.id === currentChannelState.guildId) {
      subscribeUser(newState.member.id)
    }
  })

  try {
    await discordClient.login(token)
    console.log('[discord] Bot connecting...')
  } catch (err) {
    console.error('[discord] Failed to login:', err.message)
    process.exit(1)
  }
}

async function connectToVoiceChannel(guildId, channelId) {
  if (!discordClient) throw new Error('Discord bot not initialized')
  if (!isConnected) throw new Error('Discord bot not ready')

  const { voiceConnection, voiceReceiver } = await joinDiscordVoice(discordClient, guildId, channelId)
  currentChannelState = { guildId, channelId }
  console.log('[discord] Connected to voice channel')

  initVoicePlayer(voiceConnection)

  const guild = await discordClient.guilds.fetch(guildId)
  const channel = await guild.channels.fetch(channelId)

  if (channel.type === ChannelType.GuildVoice || channel.type === ChannelType.GuildStageVoice) {
    for (const member of channel.members.values()) {
      if (!member.user.bot) subscribeUser(member.id)
    }
  }
}

function disconnectFromVoiceChannel() {
  leaveVoice()
  userBuffers.clear()
  currentChannelState = { guildId: null, channelId: null }
  console.log('[discord] Disconnected from voice channel')
}

async function sendMessage(channelId, message) {
  if (!discordClient) throw new Error('Discord bot not initialized')
  const channel = await discordClient.channels.fetch(channelId)
  const chunks = []
  for (let i = 0; i < message.length; i += 2000) chunks.push(message.slice(i, i + 2000))
  for (const chunk of chunks) await channel.send(chunk)
}

async function sendAudioToVoice(pcmData, sampleRate = 48000) {
  if (!isConnected) throw new Error('[discord] Bot not connected')
  await sendAudioToDiscord(pcmData, sampleRate)
}

async function handleJoinCommand(guildId, channelId) {
  currentChannelState = { guildId, channelId }
  await connectToVoiceChannel(guildId, channelId)
}

function getDebugState() {
  return {
    connected: isConnected,
    guildId: currentChannelState.guildId,
    channelId: currentChannelState.channelId,
    lastError,
    messageCount,
    processingQueue,
    activeListeners: [...userBuffers.keys()],
  }
}

function getDiscordClient() {
  return discordClient
}

export {
  initDiscordBot,
  connectToVoiceChannel,
  disconnectFromVoiceChannel,
  sendMessage,
  sendAudioToVoice,
  handleJoinCommand,
  getDebugState,
  getDiscordClient,
}
