import { Client, GatewayIntentBits } from 'discord.js'
import { joinVoiceChannel, EndBehaviorType, VoiceConnectionStatus, entersState, getVoiceConnection } from '@discordjs/voice'
import prism from 'prism-media'

let voiceConnection = null
let voiceReceiver = null

function createClient() {
  return new Client({
    intents: [GatewayIntentBits.Guilds, GatewayIntentBits.GuildVoiceStates, GatewayIntentBits.GuildMessages, GatewayIntentBits.MessageContent],
  })
}

function _destroyExisting(guildId) {
  const existing = getVoiceConnection(guildId)
  if (existing) {
    console.log('[discord] destroying existing voice connection for guild', guildId)
    try { existing.destroy() } catch {}
  }
  if (voiceConnection && voiceConnection !== existing) {
    try { voiceConnection.destroy() } catch {}
  }
  voiceConnection = null
  voiceReceiver = null
}

async function _tryJoin(channel, guild, attempt) {
  console.log(`[discord] join attempt ${attempt}: guild=${guild.id}, channel=${channel.id}, channelType=${channel.type}`)
  const conn = joinVoiceChannel({
    channelId: channel.id,
    guildId: guild.id,
    adapterCreator: guild.voiceAdapterCreator,
    selfDeaf: false,
    selfMute: false,
    debug: true,
  })
  console.log(`[discord] joinVoiceChannel called, waiting for Ready state...`)

  let closeCode = null
  conn.on('stateChange', (oldState, newState) => {
    if (newState.networking && newState.networking !== oldState.networking) {
      const opts = newState.networking._state?.connectionOptions ?? newState.networking.state?.connectionOptions
      if (opts) console.log('[discord] voice endpoint:', opts.endpoint, 'token:', opts.token?.slice(0,8)+'...')
      newState.networking.on('close', (evt) => {
        const code = typeof evt === 'object' ? (evt.code ?? evt) : evt
        const reason = typeof evt === 'object' ? evt.reason : ''
        closeCode = code
        console.log('[discord] voice WS closed, code:', code, 'reason:', reason?.toString?.() || '(none)')
      })
      newState.networking.on('debug', (msg) => console.log('[net]', msg.slice(0,400)))
      newState.networking.on('transitioned', (id) => console.log('[discord] transitioned, id:', id))
    }
    const oldNet = oldState.networking?.state
    const newNet = newState.networking?.state
    if (oldNet?.code !== newNet?.code) {
      const names = { 0:'OpeningWs', 1:'Identifying', 2:'UdpHandshaking', 3:'SelectingProtocol', 4:'Ready', 5:'Resuming', 6:'Closed' }
      console.log(`[discord] networking: ${names[oldNet?.code] ?? oldNet?.code ?? '?'} -> ${names[newNet?.code] ?? newNet?.code ?? '?'}`)
    }
  })

  try {
    await entersState(conn, VoiceConnectionStatus.Ready, 15_000)
    console.log('[discord] voice connection Ready')
    return conn
  } catch (err) {
    try { conn.destroy() } catch {}
    const errMsg = err.message || String(err)
    const finalErr = Object.assign(
      new Error(`Join failed after ${attempt} attempt(s): ${errMsg}${closeCode ? ` (closeCode=${closeCode})` : ''}`),
      { closeCode }
    )
    throw finalErr
  }
}

async function joinDiscordVoice(client, guildId, channelId) {
  let guild = client.guilds.cache.get(guildId)
  if (!guild) guild = await client.guilds.fetch(guildId)
  if (!guild) throw new Error(`Guild ${guildId} not found`)

  let channel = guild.channels.cache.get(channelId)
  if (!channel) {
    await guild.channels.fetch()
    channel = guild.channels.cache.get(channelId)
  }
  if (!channel) throw new Error(`Channel ${channelId} not found`)

  _destroyExisting(guildId)

  console.log('[discord] sending voice leave to clear stale session...')
  try {
    for (const shard of client.ws.shards.values()) {
      shard.send({ op: 4, d: { guild_id: guildId, channel_id: null, self_deaf: false, self_mute: false } })
    }
  } catch (e) {
    console.log('[discord] leave send error (non-fatal):', e.message)
  }
  await new Promise(r => {
    const botId = client.user?.id
    const onVoiceState = (oldState, newState) => {
      if (newState.guild?.id === guildId && newState.channelId === null && (!botId || newState.member?.user?.id === botId || newState.member?.id === botId)) {
        client.off('voiceStateUpdate', onVoiceState)
        clearTimeout(timer)
        console.log('[discord] voice leave confirmed by Discord')
        r()
      }
    }
    const timer = setTimeout(() => {
      client.off('voiceStateUpdate', onVoiceState)
      console.log('[discord] voice leave not confirmed, proceeding after timeout')
      r()
    }, 5000)
    client.on('voiceStateUpdate', onVoiceState)
  })

  for (let attempt = 1; attempt <= 3; attempt++) {
    try {
      voiceConnection = await _tryJoin(channel, guild, attempt)
      voiceReceiver = voiceConnection.receiver

      voiceConnection.on(VoiceConnectionStatus.Disconnected, () => {
        console.log('[discord] voice disconnected, voice.js will attempt rejoin')
      })

      console.log('[discord] ✓ Voice connection successful!')
      return { voiceConnection, voiceReceiver }
    } catch (err) {
      console.log(`[discord] attempt ${attempt} failed: ${err.message}, closeCode=${err.closeCode}`)
      _destroyExisting(guildId)

      // Fail fast on 4017 - this is a permanent auth issue
      if (err.closeCode === 4017) {
        throw new Error(`Discord rejected voice connection (error 4017: Invalid Guild). ` +
          `This means: bot is not invited to the server, lacks voice permissions, ` +
          `or the voice channel doesn't exist. Verify bot permissions in Discord.`)
      }

      const delay = err.closeCode === 4006 ? 8000 : 4000
      if (attempt < 3) {
        console.log(`[discord] waiting ${delay}ms before retry (attempt ${attempt}/3)...`)
        await new Promise(r => setTimeout(r, delay))
        if (err.closeCode === 4006) {
          console.log('[discord] 4006: stale session — clearing and retrying...')
          try {
            for (const shard of client.ws.shards.values()) {
              shard.send({ op: 4, d: { guild_id: guildId, channel_id: null, self_deaf: false, self_mute: false } })
            }
          } catch (e) { console.log('[discord] leave send error:', e.message) }
          await new Promise(r => setTimeout(r, 2000))
        }
      }
    }
  }

  throw new Error('Voice connection failed after 3 attempts')
}

function subscribeToSpeaker(userId, onPcmChunk) {
  if (!voiceReceiver) return null

  const existing = voiceReceiver.subscriptions.get(userId)
  if (existing) return existing

  const stream = voiceReceiver.subscribe(userId, {
    end: { behavior: EndBehaviorType.Manual },
  })

  const decoder = new prism.opus.Decoder({ frameSize: 960, channels: 2, rate: 48000 })
  stream.pipe(decoder)

  decoder.on('data', (pcmBuf) => {
    const i16 = new Int16Array(pcmBuf.buffer, pcmBuf.byteOffset, pcmBuf.byteLength / 2)
    const f32 = new Float32Array(i16.length)
    for (let i = 0; i < i16.length; i++) f32[i] = i16[i] / 32768
    onPcmChunk(userId, f32)
  })

  decoder.on('error', () => {})
  stream.on('close', () => decoder.destroy())

  return stream
}

function leaveVoice() {
  if (voiceConnection) {
    voiceConnection.destroy()
    voiceConnection = null
    voiceReceiver = null
  }
}

export { createClient, joinDiscordVoice, subscribeToSpeaker, leaveVoice }
