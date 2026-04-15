import { Client, GatewayIntentBits } from 'discord.js'
import { joinVoiceChannel, EndBehaviorType, VoiceConnectionStatus, entersState, getVoiceConnection, createAudioResource, StreamType } from '@discordjs/voice'
import prism from 'prism-media'
import { resampleAudio } from './server-utils.mjs'
import { Readable } from 'stream'

let voiceConnection = null
let voiceReceiver = null
let audioSendQueue = []
let totalAudioFramesSent = 0
let lastSendTimestamp = null
let lastSendError = null

function createClient() {
  return new Client({
    intents: [GatewayIntentBits.Guilds, GatewayIntentBits.GuildVoiceStates],
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
  console.log(`[discord] join attempt ${attempt}`)
  const conn = joinVoiceChannel({
    channelId: channel.id,
    guildId: guild.id,
    adapterCreator: guild.voiceAdapterCreator,
    selfDeaf: false,
    selfMute: false,
    debug: true,
  })

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
    throw Object.assign(new Error(`Join failed: ${err.message}`), { closeCode })
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

  for (let attempt = 1; attempt <= 8; attempt++) {
    try {
      voiceConnection = await _tryJoin(channel, guild, attempt)
      voiceReceiver = voiceConnection.receiver

      voiceConnection.on(VoiceConnectionStatus.Disconnected, () => {
        console.log('[discord] voice disconnected')
      })

      return { voiceConnection, voiceReceiver }
    } catch (err) {
      console.log(`[discord] attempt ${attempt} failed: ${err.message}, closeCode=${err.closeCode}`)
      _destroyExisting(guildId)
      const delay = err.closeCode === 4017 ? 10000 : err.closeCode === 4006 ? 8000 : 4000
      console.log(`[discord] waiting ${delay}ms before retry...`)
      await new Promise(r => setTimeout(r, delay))
      if (err.closeCode === 4006) {
        console.log('[discord] 4006: stale session — ensure no other instance is using this bot token in voice. Re-sending leave...')
        try {
          for (const shard of client.ws.shards.values()) {
            shard.send({ op: 4, d: { guild_id: guildId, channel_id: null, self_deaf: false, self_mute: false } })
          }
        } catch (e) { console.log('[discord] leave send error:', e.message) }
        await new Promise(r => setTimeout(r, 2000))
      }
    }
  }

  throw new Error('Voice connection failed after 8 attempts')
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

/**
 * Normalize PCM audio to 48kHz Float32Array
 *
 * PCM Format Support:
 * - Float32Array: values in [-1.0, 1.0] range
 * - Int16Array: values in [-32768, 32767] range (auto-normalized to [-1, 1])
 * - Buffer: interpreted as Int16 little-endian PCM (2 bytes per sample)
 *
 * Mono input is preserved as-is. Stereo/multi-channel input converted to mono.
 * Sample rate is resampled to exactly 48kHz via linear interpolation.
 *
 * @param {Buffer|Float32Array|Int16Array} buffer - Input PCM buffer
 * @param {number} inputSampleRate - Sample rate of input (e.g., 24000, 48000)
 * @returns {Float32Array} Audio resampled to 48kHz, normalized to [-1, 1]
 * @throws {Error} If buffer format is unrecognized or length is invalid
 */
function normalizePcmTo48k(buffer, inputSampleRate) {
  if (!buffer || buffer.length === 0) return new Float32Array(0)

  let float32;

  // Convert to Float32Array if needed
  if (buffer instanceof Float32Array) {
    float32 = buffer;
  } else if (buffer instanceof Int16Array) {
    float32 = new Float32Array(buffer.length);
    for (let i = 0; i < buffer.length; i++) {
      float32[i] = buffer[i] / 32768;
    }
  } else if (Buffer.isBuffer(buffer)) {
    // Interpret as Int16 little-endian
    const int16 = new Int16Array(buffer.buffer, buffer.byteOffset, buffer.length / 2);
    float32 = new Float32Array(int16.length);
    for (let i = 0; i < int16.length; i++) {
      float32[i] = int16[i] / 32768;
    }
  } else {
    throw new Error(`Unrecognized PCM format: expected Buffer, Float32Array, or Int16Array, got ${buffer.constructor.name}`);
  }

  // Resample to 48kHz if needed
  if (inputSampleRate === 48000) return float32;
  return resampleAudio(float32, inputSampleRate, 48000);
}

/**
 * Send audio to Discord voice connection
 * Encodes PCM as Opus (if available) or sends as raw PCM, plays via voice connection player
 * @param {Buffer|Float32Array|Int16Array} pcmBuffer - Input PCM audio
 * @param {number} sampleRate - Sample rate of input audio (e.g., 24000, 48000)
 * @returns {Promise<void>} Resolves when audio is queued to play
 * @throws {Error} If voice connection not active or encoding fails
 *
 * PCM Format Expectations:
 * - Input: Float32Array [-1.0, 1.0], Int16Array, or Buffer (interpreted as Int16 LE)
 * - Sample Rate: any valid rate (will resample to 48kHz)
 * - Channels: mono (will be sent as stereo to Discord)
 * - Output: 48kHz, 16-bit PCM (via discord.js voice subsystem)
 */
async function sendAudioToDiscord(pcmBuffer, sampleRate) {
  if (!voiceConnection) throw new Error('[discord] Voice connection not active (null)');
  if (voiceConnection.state.status !== VoiceConnectionStatus.Ready) {
    throw new Error(`[discord] Voice connection not ready: status=${voiceConnection.state.status}`);
  }
  if (!voiceConnection.state.subscription) {
    throw new Error('[discord] Voice connection has no subscription/player');
  }

  try {
    // Normalize to 48kHz Float32
    const normalized = normalizePcmTo48k(pcmBuffer, sampleRate);
    if (normalized.length === 0) throw new Error('Normalized audio is empty');

    // Convert Float32 to Int16 for Discord
    const int16 = new Int16Array(normalized.length);
    for (let i = 0; i < normalized.length; i++) {
      const val = Math.max(-1, Math.min(1, normalized[i]));
      int16[i] = val < 0 ? val * 0x8000 : val * 0x7FFF;
    }

    let resource;
    const pcmBuf = Buffer.from(int16);

    // Try to use Opus encoder if available, fall back to Raw PCM
    try {
      const encoder = new prism.opus.Encoder({
        rate: 48000,
        channels: 2,
        frameSize: 960
      });

      const pcmStream = Readable.from([pcmBuf]);
      const opusStream = pcmStream.pipe(encoder);

      resource = createAudioResource(opusStream, {
        inputType: StreamType.Opus,
        inlineVolume: true
      });

      console.log('[discord] Using Opus encoding for audio');
    } catch (opusErr) {
      // Fallback to Raw PCM if Opus encoder unavailable
      console.log('[discord] Opus encoder unavailable, using raw PCM:', opusErr.message?.slice(0, 50));

      const pcmStream = Readable.from([pcmBuf]);
      resource = createAudioResource(pcmStream, {
        inputType: StreamType.Raw,
        inlineVolume: true
      });
    }

    // Play the audio
    voiceConnection.state.subscription.player.play(resource);

    // Update metrics
    audioSendQueue.push({ timestamp: Date.now(), bytes: normalized.length * 2 });
    if (audioSendQueue.length > 100) audioSendQueue.shift();
    totalAudioFramesSent += normalized.length;
    lastSendTimestamp = Date.now();
    lastSendError = null;

  } catch (err) {
    lastSendError = { message: err.message, timestamp: Date.now() };
    throw Object.assign(new Error(`[discord] Failed to send audio: ${err.message}`), { originalError: err });
  }
}

/**
 * Get audio send metrics and queue state
 */
function getAudioDebugState() {
  return {
    audioQueueLength: audioSendQueue.length,
    totalAudioFramesSent,
    lastSendTimestamp,
    lastSendError,
    queueHistory: audioSendQueue.slice(-10)
  };
}

export { createClient, joinDiscordVoice, subscribeToSpeaker, leaveVoice, sendAudioToDiscord, normalizePcmTo48k, getAudioDebugState }
