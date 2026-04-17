import { ChannelType } from 'discord.js'
import { createClient, joinDiscordVoice, subscribeToSpeaker, leaveVoice, lastVoiceCloseCode } from './discord-bot-client.js'
import { initVoicePlayer } from 'dispipe/voice'
import { onPcmChunk, init as initVad, getBuffers } from './discord-vad.js'

let discordClient = null
let isConnected = false
let currentChannelState = { guildId: null, channelId: null }
const lastError = { value: null }
let messageCount = 0
const processingQueue = []
let _onCommand = null

initVad(processingQueue, lastError)

async function initDiscordBot(onUserAudio, onCommand, onReady) {
  const token = process.env.DISCORD_TOKEN || process.env.DISCORD_BOT_TOKEN
  if (!token) { console.log('[discord] DISCORD_TOKEN not set, Discord bot disabled'); return }

  _onCommand = onCommand
  discordClient = createClient()

  let onReadyCalled = false
  discordClient.on('ready', () => {
    console.log('[discord] ✓ Bot ready - logged in as', discordClient.user.tag, `(ID: ${discordClient.user.id})`)
    isConnected = true
    if (onReady && !onReadyCalled) { onReadyCalled = true; onReady() }
  })

  discordClient.on('error', (err) => {
    console.error('[discord] Client error:', err.message)
    lastError.value = { message: err.message, timestamp: Date.now() }
  })

  discordClient.on('warn', (warn) => console.warn('[discord] Client warning:', warn))

  discordClient.on('messageCreate', async (message) => {
    if (message.author.bot || !message.guild) return

    if (message.content.startsWith('!join ')) {
      const channelId = message.content.slice(6).trim()
      if (!channelId) { await message.reply('Usage: !join <channel-id>'); return }
      try {
        await handleJoinCommand(message.guildId, channelId)
        await message.reply(`Joining voice channel ${channelId}...`)
      } catch (err) { await message.reply('Error joining channel: ' + err.message) }
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
      } catch (err) { await message.reply('Error: ' + err.message) }
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

  voiceReceiver.speaking.on('start', (userId) => {
    subscribeToSpeaker(userId, onPcmChunk)
    console.log(`[voice] subscribed to speaker ${userId}`)
  })

  const guild = await discordClient.guilds.fetch(guildId)
  const channel = await guild.channels.fetch(channelId)
  if (channel.type === ChannelType.GuildVoice || channel.type === ChannelType.GuildStageVoice) {
    for (const member of channel.members.values()) {
      if (!member.user.bot) {
        subscribeToSpeaker(member.id, onPcmChunk)
        console.log(`[voice] pre-subscribed to ${member.id}`)
      }
    }
  }
}

function disconnectFromVoiceChannel() {
  leaveVoice()
  getBuffers().clear()
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

async function handleJoinCommand(guildId, channelId) {
  currentChannelState = { guildId, channelId }
  await connectToVoiceChannel(guildId, channelId)
}

function getDebugState() {
  return {
    connected: isConnected,
    guildId: currentChannelState.guildId,
    channelId: currentChannelState.channelId,
    lastError: lastError.value,
    lastVoiceCloseCode: lastVoiceCloseCode.value,
    lastVoiceCloseReason: lastVoiceCloseCode.reason,
    messageCount,
    processingQueue,
    activeListeners: [...getBuffers().keys()],
  }
}

function getDiscordClient() { return discordClient }

export {
  initDiscordBot,
  connectToVoiceChannel,
  disconnectFromVoiceChannel,
  sendMessage,
  handleJoinCommand,
  getDebugState,
  getDiscordClient,
}
