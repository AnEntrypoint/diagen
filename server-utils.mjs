const ARKIT_NAMES = [
  'browInnerUp', 'browDownLeft', 'browDownRight', 'browOuterUpLeft', 'browOuterUpRight',
  'eyeLookUpLeft', 'eyeLookUpRight', 'eyeLookDownLeft', 'eyeLookDownRight',
  'eyeLookInLeft', 'eyeLookInRight', 'eyeLookOutLeft', 'eyeLookOutRight',
  'eyeBlinkLeft', 'eyeBlinkRight', 'eyeSquintLeft', 'eyeSquintRight',
  'eyeWideLeft', 'eyeWideRight', 'cheekPuff', 'cheekSquintLeft', 'cheekSquintRight',
  'noseSneerLeft', 'noseSneerRight', 'jawOpen', 'jawForward', 'jawLeft', 'jawRight',
  'mouthFunnel', 'mouthPucker', 'mouthLeft', 'mouthRight',
  'mouthRollUpper', 'mouthRollLower', 'mouthShrugUpper', 'mouthShrugLower',
  'mouthOpen', 'mouthClose', 'mouthSmileLeft', 'mouthSmileRight',
  'mouthFrownLeft', 'mouthFrownRight', 'mouthDimpleLeft', 'mouthDimpleRight',
  'mouthUpperUpLeft', 'mouthUpperUpRight', 'mouthLowerDownLeft', 'mouthLowerDownRight',
  'mouthPressLeft', 'mouthPressRight', 'mouthStretchLeft', 'mouthStretchRight'
]

function encodeWAV(float32Data, sampleRate) {
  const wavBuf = new ArrayBuffer(44 + float32Data.length * 2)
  const view = new DataView(wavBuf)
  const writeStr = (o, s) => { for (let i = 0; i < s.length; i++) view.setUint8(o + i, s.charCodeAt(i)) }
  writeStr(0, 'RIFF')
  view.setUint32(4, 36 + float32Data.length * 2, true)
  writeStr(8, 'WAVE')
  writeStr(12, 'fmt ')
  view.setUint32(16, 16, true)
  view.setUint16(20, 1, true)
  view.setUint16(22, 1, true)
  view.setUint32(24, sampleRate, true)
  view.setUint32(28, sampleRate * 2, true)
  view.setUint16(32, 2, true)
  view.setUint16(34, 16, true)
  writeStr(36, 'data')
  view.setUint32(40, float32Data.length * 2, true)
  for (let i = 0; i < float32Data.length; i++) {
    const s = Math.max(-1, Math.min(1, float32Data[i]))
    view.setInt16(44 + i * 2, s < 0 ? s * 0x8000 : s * 0x7FFF, true)
  }
  return Buffer.from(wavBuf)
}

function resampleAudio(float32Data, fromRate, toRate) {
  const ratio = fromRate / toRate
  const newLen = Math.round(float32Data.length / ratio)
  const result = new Float32Array(newLen)
  for (let i = 0; i < newLen; i++) {
    const idx = i * ratio
    const lo = Math.floor(idx)
    const hi = Math.min(lo + 1, float32Data.length - 1)
    const frac = idx - lo
    result[i] = float32Data[lo] * (1 - frac) + float32Data[hi] * frac
  }
  return result
}

function buildAfan(frames, fps = 30) {
  const numBlendshapes = ARKIT_NAMES.length
  const numFrames = frames.length
  const totalSize = 12 + (numFrames * numBlendshapes)
  const buf = Buffer.alloc(totalSize)
  let offset = 0
  buf.writeUInt32LE(0x4146414E, offset); offset += 4
  buf.writeUInt8(2, offset); offset += 1
  buf.writeUInt8(fps, offset); offset += 1
  buf.writeUInt8(numBlendshapes, offset); offset += 1
  buf.writeUInt8(0, offset); offset += 1
  buf.writeUInt32LE(numFrames, offset); offset += 4
  for (const frame of frames) {
    for (let i = 0; i < numBlendshapes; i++) {
      buf[offset++] = Math.round(Math.max(0, Math.min(1, frame[ARKIT_NAMES[i]] || 0)) * 255)
    }
  }
  return buf
}

export { ARKIT_NAMES, encodeWAV, resampleAudio, buildAfan }
