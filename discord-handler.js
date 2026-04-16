import { ChannelType } from 'discord.js'
import { createClient, joinDiscordVoice, subscribeToSpeaker, leaveVoice, sendAudioToDiscord, getAudioDebugState } from './discord-bot-client.js'
import { processUserAudio } from './discord-voice-processor.js'

let discordClient = null
let isConnected = false
let currentChannelState = { guildId: null, channelId: null }
let lastError = null
let messageCount = 0
let processingQueue = []

/**
 * Initialize the Discord bot and register event handlers
 * @param {Function} onUserAudio - Callback when user audio is received: (userId, pcmChunk) => void
 * @param {Function} onCommand - Callback when command message received: (userId, content) => Promise<string>
 * @returns {Promise<void>}
 */
async function initDiscordBot(onUserAudio, onCommand) {
  const token = process.env.DISCORD_TOKEN || process.env.DISCORD_BOT_TOKEN
  if (!token) {
    console.log('[discord] DISCORD_TOKEN not set, Discord bot disabled')
    return
  }

  discordClient = createClient()

  discordClient.on('ready', () => {
    console.log('[discord] Logged in as', discordClient.user.tag)
    isConnected = true
  })

  discordClient.on('messageCreate', async (message) => {
    // Ignore bot's own messages and DMs
    if (message.author.bot || !message.guild) return

    // !join <channel-id> command
    if (message.content.startsWith('!join ')) {
      const channelId = message.content.slice(6).trim()
      if (!channelId) {
        await message.reply('Usage: !join <channel-id>')
        return
      }

      try {
        await handleJoinCommand(message.guildId, channelId)
        await message.reply(`Joining voice channel ${channelId}...`)
      } catch (err) {
        console.error('[discord] Join command error:', err)
        await message.reply('Error joining channel: ' + err.message)
      }
    }

    // Simple command handling: !diagen <prompt>
    if (message.content.startsWith('!diagen ')) {
      const prompt = message.content.slice(8).trim()
      if (!prompt) {
        await message.reply('Usage: !diagen <prompt>')
        return
      }

      try {
        await message.channel.sendTyping()
        const response = await onCommand(message.author.id, prompt)

        // Split response into chunks if too long
        const chunks = []
        for (let i = 0; i < response.length; i += 2000) {
          chunks.push(response.slice(i, i + 2000))
        }

        for (const chunk of chunks) {
          await message.reply(chunk)
        }
      } catch (err) {
        console.error('[discord] Command error:', err)
        await message.reply('Error processing command: ' + err.message)
      }
    }
  })

  discordClient.on('voiceStateUpdate', (oldState, newState) => {
    // Bot joins when it detects voice activity
    if (newState.channelId && newState.member?.user?.bot === false) {
      console.log('[discord]', newState.member.user.username, 'joined voice channel')
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

/**
 * Connect the bot to a voice channel and start listening
 * @param {string} guildId - Discord guild ID
 * @param {string} channelId - Discord voice channel ID
 * @returns {Promise<void>}
 */
async function connectToVoiceChannel(guildId, channelId) {
  if (!discordClient) throw new Error('Discord bot not initialized')
  if (!isConnected) throw new Error('Discord bot not ready')

  try {
    const { voiceReceiver } = await joinDiscordVoice(discordClient, guildId, channelId)
    console.log('[discord] Connected to voice channel')

    // Set up speaker subscriptions for all users in the channel
    const guild = await discordClient.guilds.fetch(guildId)
    const channel = await guild.channels.fetch(channelId)

    if (channel.type === ChannelType.GuildVoice) {
      for (const member of channel.members.values()) {
        if (!member.user.bot) {
          subscribeToSpeaker(member.id, async (userId, pcmChunk) => {
            try {
              const audioOutput = await processUserAudio(pcmChunk, 48000, userId)
              await sendAudioToDiscord(audioOutput, 48000)
              console.log(`[discord] Processed audio for ${userId}: ${audioOutput.length} bytes`)
            } catch (err) {
              console.error(`[discord] Audio processing error for userId=${userId}: ${err.message}`)
            }
            if (onUserAudio) onUserAudio(userId, pcmChunk)
          })
        }
      }
    }
  } catch (err) {
    console.error('[discord] Failed to connect to voice:', err)
    throw err
  }
}

/**
 * Disconnect from the current voice channel
 */
function disconnectFromVoiceChannel() {
  leaveVoice()
  console.log('[discord] Disconnected from voice channel')
}

/**
 * Send a text message to a Discord channel
 * @param {string} channelId - Discord channel ID
 * @param {string} message - Message content
 * @returns {Promise<void>}
 */
async function sendMessage(channelId, message) {
  if (!discordClient) throw new Error('Discord bot not initialized')

  try {
    const channel = await discordClient.channels.fetch(channelId)
    const chunks = []
    for (let i = 0; i < message.length; i += 2000) {
      chunks.push(message.slice(i, i + 2000))
    }
    for (const chunk of chunks) {
      await channel.send(chunk)
    }
  } catch (err) {
    console.error('[discord] Failed to send message:', err)
    throw err
  }
}

/**
 * Send audio to Discord voice channel
 * @param {Buffer|Float32Array|Int16Array} pcmData - PCM audio data
 * @param {number} sampleRate - Sample rate of input (default 48000)
 * @returns {Promise<void>}
 */
async function sendAudioToVoice(pcmData, sampleRate = 48000) {
  if (!isConnected) {
    throw new Error('[discord] Bot not connected to Discord');
  }
  try {
    await sendAudioToDiscord(pcmData, sampleRate);
  } catch (err) {
    console.error(`[discord] Failed to send audio: ${err.message}`);
    lastError = { message: err.message, timestamp: Date.now() };
    throw err;
  }
}

/**
 * Check if bot is connected to Discord
 */
function isDiscordConnected() {
  return isConnected
}

/**
 * Get the Discord client instance
 */
function getDiscordClient() {
  return discordClient
}

/**
 * Handle !join command - store channel state and connect
 * @param {string} guildId - Discord guild ID
 * @param {string} channelId - Discord voice channel ID
 * @returns {Promise<void>}
 */
async function handleJoinCommand(guildId, channelId) {
  currentChannelState.guildId = guildId
  currentChannelState.channelId = channelId
  console.log('[discord] Stored channel state:', currentChannelState)
  await connectToVoiceChannel(guildId, channelId)
}

/**
 * Get current stored channel state
 * @returns {Object} Copy of current channel state
 */
function getCurrentChannelState() {
  return { ...currentChannelState }
}

let onUserAudio = null
let onCommand = null

/**
 * Get debug state for observability endpoint
 */
function getDebugState() {
  const audioState = getAudioDebugState();
  return {
    connected: isConnected,
    guildId: currentChannelState.guildId,
    channelId: currentChannelState.channelId,
    lastError: lastError,
    messageCount: messageCount,
    processingQueue: processingQueue,
    audio: audioState,
  }
}

export {
  initDiscordBot,
  connectToVoiceChannel,
  disconnectFromVoiceChannel,
  sendMessage,
  sendAudioToVoice,
  isDiscordConnected,
  getDiscordClient,
  handleJoinCommand,
  getCurrentChannelState,
  getDebugState,
}
