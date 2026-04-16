import { createAudioPlayer, createAudioResource, StreamType, AudioPlayerStatus, VoiceConnectionStatus } from '@discordjs/voice'
import { PassThrough } from 'node:stream'
import prism from 'prism-media'

let audioPlayer = null
let pcmInput = null
let voiceConn = null
let _pushCount = 0
const FRAME = 960 * 2 * 2
let _accumBuf = Buffer.alloc(0)

function _makeStream() {
  if (pcmInput) {
    try { pcmInput.destroy() } catch {}
  }
  pcmInput = new PassThrough({ highWaterMark: 960 * 2 * 2 * 200 })
  const encoder = new prism.opus.Encoder({ rate: 48000, channels: 2, frameSize: 960 })
  pcmInput.pipe(encoder)
  const resource = createAudioResource(encoder, { inputType: StreamType.Opus })
  return resource
}

function initVoicePlayer(connection) {
  if (audioPlayer) { try { audioPlayer.stop() } catch {} }
  voiceConn = connection
  audioPlayer = createAudioPlayer()

  console.log('[voice] connection state on init:', connection.state.status)

  connection.on('stateChange', (oldState, newState) => {
    console.log(`[voice] connection: ${oldState.status} -> ${newState.status}`)
    if (newState.status === VoiceConnectionStatus.Disconnected) {
      try { connection.rejoin() } catch {}
    }
  })

  audioPlayer.on('error', (err) => console.error('[voice] player error:', err.message))
  audioPlayer.on('stateChange', (oldState, newState) => {
    console.log(`[voice] player: ${oldState.status} -> ${newState.status}`)
    if (newState.status === AudioPlayerStatus.Idle && oldState.status !== AudioPlayerStatus.Idle) {
      console.log('[voice] stream ended, restarting stream')
      const resource = _makeStream()
      audioPlayer.play(resource)
    }
  })

  connection.subscribe(audioPlayer)

  const resource = _makeStream()
  audioPlayer.play(resource)
  console.log('[voice] player started')
}

function pushAudioFrame(monoBuffer) {
  if (!pcmInput || pcmInput.destroyed) return
  const src = monoBuffer instanceof Float32Array ? monoBuffer : (() => {
    const i16 = new Int16Array(monoBuffer.buffer || monoBuffer, monoBuffer.byteOffset || 0, (monoBuffer.byteLength || monoBuffer.length) / 2)
    const f = new Float32Array(i16.length)
    for (let i = 0; i < i16.length; i++) f[i] = i16[i] / 32768
    return f
  })()
  const stereo = new Int16Array(src.length * 2)
  for (let i = 0; i < src.length; i++) {
    const s = Math.max(-1, Math.min(1, src[i]))
    const v = s < 0 ? s * 32768 : s * 32767
    stereo[i * 2] = v
    stereo[i * 2 + 1] = v
  }
  const s16Buf = Buffer.from(stereo.buffer, stereo.byteOffset, stereo.byteLength)
  if (_accumBuf.length === 0 && s16Buf.length >= FRAME) {
    let off = 0
    while (off + FRAME <= s16Buf.length) { pcmInput.write(s16Buf.subarray(off, off + FRAME)); off += FRAME; _pushCount++ }
    if (off < s16Buf.length) _accumBuf = Buffer.from(s16Buf.subarray(off))
  } else {
    _accumBuf = Buffer.concat([_accumBuf, s16Buf])
  }
  while (_accumBuf.length >= FRAME) {
    pcmInput.write(_accumBuf.subarray(0, FRAME))
    _accumBuf = _accumBuf.subarray(FRAME)
    _pushCount++
    if (_pushCount <= 5 || _pushCount % 500 === 0) {
      let peak = 0; for (let i = 0; i < src.length; i++) { const v = Math.abs(src[i]); if (v > peak) peak = v }
      console.log(`[voice] push #${_pushCount}, bytes=${FRAME}, peak=${peak.toFixed(4)}, player=${audioPlayer?.state?.status}`)
    }
  }
}

function stopAudio() {
  if (pcmInput) {
    try { pcmInput.end() } catch {}
    pcmInput = null
  }
  if (audioPlayer) {
    audioPlayer.stop()
    audioPlayer = null
  }
  voiceConn = null
}

export { initVoicePlayer, pushAudioFrame, stopAudio }
