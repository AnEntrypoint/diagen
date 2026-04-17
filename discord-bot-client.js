import { Client, GatewayIntentBits } from 'discord.js'
import { joinVoiceChannel, EndBehaviorType, VoiceConnectionStatus, entersState, getVoiceConnection } from '@discordjs/voice'
import prism from 'prism-media'

let voiceConnection = null
let voiceReceiver = null
let _client = null
export const lastVoiceCloseCode = { value: null, reason: null }

function createClient() {
  return new Client({
    intents: [GatewayIntentBits.Guilds, GatewayIntentBits.GuildVoiceStates],
  })
}

function _destroyExisting(guildId) {
  const existing = getVoiceConnection(guildId)
  if (existing) try { existing.destroy() } catch {}
  if (voiceConnection && voiceConnection !== existing) try { voiceConnection.destroy() } catch {}
  voiceConnection = null
  voiceReceiver = null
}

async function _tryJoin(channel, guild) {
  const conn = joinVoiceChannel({
    channelId: channel.id,
    guildId: guild.id,
    adapterCreator: guild.voiceAdapterCreator,
    selfDeaf: false,
    selfMute: false,
    debug: false,
    daveEncryption: false,
  })

  conn.addStatePacket = (packet) => {
    console.log(`[discord] VOICE_STATE_UPDATE session_id=${packet.session_id} channel_id=${packet.channel_id}`)
    conn.packets.state = packet
    if (packet.self_deaf !== undefined) conn.joinConfig.selfDeaf = packet.self_deaf
    if (packet.self_mute !== undefined) conn.joinConfig.selfMute = packet.self_mute
    if (packet.channel_id) conn.joinConfig.channelId = packet.channel_id
  }

  const origAddServerPacket = conn.addServerPacket.bind(conn)
  conn.addServerPacket = (packet) => {
    conn.packets.server = packet
    if (packet.endpoint) {
      console.log(`[discord] VOICE_SERVER_UPDATE endpoint=${packet.endpoint}, waiting 600ms for fresh VOICE_STATE_UPDATE...`)
      setTimeout(() => {
        console.log(`[discord] configureNetworking session_id=${conn.packets.state?.session_id}`)
        conn.configureNetworking()
      }, 600)
    } else {
      origAddServerPacket(packet)
    }
  }

  const netNames = { 0:'OpeningWs', 1:'Identifying', 2:'UdpHandshaking', 3:'SelectingProtocol', 4:'Ready', 5:'Resuming', 6:'Closed' }
  let closeCode = null
  conn.on('stateChange', (oldState, newState) => {
    if (newState.networking && newState.networking !== oldState.networking) {
      newState.networking.on('close', (evt) => {
        const code = typeof evt === 'object' ? (evt.code ?? evt) : evt
        const reason = typeof evt === 'object' ? evt.reason : ''
        closeCode = code; lastVoiceCloseCode.value = code; lastVoiceCloseCode.reason = reason?.toString?.() || ''
        console.log('[discord] voice WS closed, code:', code)
      })
    }
    const oldCode = oldState.networking?.state?.code, newCode = newState.networking?.state?.code
    if (oldCode !== newCode) console.log(`[discord] networking: ${netNames[oldCode] ?? '?'} -> ${netNames[newCode] ?? '?'}`)
  })

  try {
    await entersState(conn, VoiceConnectionStatus.Ready, 20_000)
    console.log('[discord] voice connection Ready')
    return conn
  } catch (err) {
    try { conn.destroy() } catch {}
    throw Object.assign(new Error(err.message || String(err)), { closeCode })
  }
}

async function _leaveAndWaitChannelNull(guildId, waitMs = 6000) {
  _destroyExisting(guildId)
  const client = _client
  const botId = client?.user?.id
  try { await client.rest.patch(`/guilds/${guildId}/members/@me`, { body: { channel_id: null } }) } catch(e) {}
  try { for (const shard of client.ws.shards.values()) shard.send({ op: 4, d: { guild_id: guildId, channel_id: null, self_deaf: false, self_mute: false } }) } catch(e) {}
  await new Promise(r => {
    const onVoiceState = (oldState, newState) => {
      const memberId = newState.member?.user?.id ?? newState.member?.id
      if (newState.guild?.id === guildId && newState.channelId === null && (!botId || memberId === botId)) {
        client.off('voiceStateUpdate', onVoiceState); clearTimeout(timer)
        console.log('[discord] leave confirmed, fresh session_id will come on rejoin')
        r()
      }
    }
    const timer = setTimeout(() => { client.off('voiceStateUpdate', onVoiceState); console.log('[discord] leave wait timeout'); r() }, waitMs)
    client.on('voiceStateUpdate', onVoiceState)
  })
  await new Promise(r => setTimeout(r, 500))
}

async function joinDiscordVoice(client, guildId, channelId) {
  _client = client
  let guild = client.guilds.cache.get(guildId) || await client.guilds.fetch(guildId)
  if (!guild) throw new Error(`Guild ${guildId} not found`)

  let channel = guild.channels.cache.get(channelId)
  if (!channel) { await guild.channels.fetch(); channel = guild.channels.cache.get(channelId) }
  if (!channel) throw new Error(`Channel ${channelId} not found`)

  const me = guild.members.cache.get(client.user.id) || await guild.members.fetch(client.user.id)
  const perms = channel.permissionsFor(me)
  if (perms && !perms.has('Connect')) throw new Error(`Bot lacks CONNECT permission in channel ${channelId}`)

  await _leaveAndWaitChannelNull(guildId)

  for (let attempt = 1; attempt <= 5; attempt++) {
    console.log(`[discord] join attempt ${attempt}: guild=${guild.id} channel=${channel.id}`)
    try {
      voiceConnection = await _tryJoin(channel, guild)
      voiceReceiver = voiceConnection.receiver
      voiceConnection.on(VoiceConnectionStatus.Disconnected, () => console.log('[discord] voice disconnected'))
      console.log('[discord] ✓ Voice connection established!')
      return { voiceConnection, voiceReceiver }
    } catch (err) {
      console.log(`[discord] attempt ${attempt} failed: ${err.message} closeCode=${err.closeCode}`)
      if (attempt === 5) throw new Error(`Voice join failed after 5 attempts: ${err.message}`)
      console.log(`[discord] leaving and waiting for fresh session_id...`)
      await _leaveAndWaitChannelNull(guildId)
      guild = client.guilds.cache.get(guildId) || await client.guilds.fetch(guildId)
      channel = guild.channels.cache.get(channelId)
      if (!channel) { await guild.channels.fetch(); channel = guild.channels.cache.get(channelId) }
    }
  }
}

function subscribeToSpeaker(userId, onPcmChunk) {
  if (!voiceReceiver) return null
  const existing = voiceReceiver.subscriptions.get(userId)
  if (existing) return existing
  const stream = voiceReceiver.subscribe(userId, { end: { behavior: EndBehaviorType.Manual } })
  const decoder = new prism.opus.Decoder({ frameSize: 960, channels: 2, rate: 48000 })
  stream.pipe(decoder)
  decoder.on('data', (pcmBuf) => {
    const i16 = new Int16Array(pcmBuf.buffer, pcmBuf.byteOffset, pcmBuf.byteLength / 2)
    const monoLen = i16.length / 2
    const f32 = new Float32Array(monoLen)
    for (let i = 0; i < monoLen; i++) f32[i] = (i16[i * 2] + i16[i * 2 + 1]) / 2 / 32768
    onPcmChunk(userId, f32)
  })
  decoder.on('error', () => {})
  stream.on('close', () => decoder.destroy())
  return stream
}

function leaveVoice() {
  if (voiceConnection) { voiceConnection.destroy(); voiceConnection = null; voiceReceiver = null }
}

export { createClient, joinDiscordVoice, subscribeToSpeaker, leaveVoice }
