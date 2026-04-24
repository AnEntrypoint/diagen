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
  const numSamples = float32Data.length
  const dataBytes = numSamples * 2
  const buf = Buffer.alloc(44 + dataBytes)
  buf.write('RIFF', 0, 'ascii')
  buf.writeUInt32LE(36 + dataBytes, 4)
  buf.write('WAVEfmt ', 8, 'ascii')
  buf.writeUInt32LE(16, 16); buf.writeUInt16LE(1, 20); buf.writeUInt16LE(1, 22)
  buf.writeUInt32LE(sampleRate, 24); buf.writeUInt32LE(sampleRate * 2, 28)
  buf.writeUInt16LE(2, 32); buf.writeUInt16LE(16, 34)
  buf.write('data', 36, 'ascii')
  buf.writeUInt32LE(dataBytes, 40)
  const samples = new Int16Array(buf.buffer, buf.byteOffset + 44, numSamples)
  for (let i = 0; i < numSamples; i++) {
    const s = float32Data[i]
    const clamped = s > 1 ? 1 : s < -1 ? -1 : s
    samples[i] = clamped < 0 ? clamped * 0x8000 : clamped * 0x7FFF
  }
  return buf
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
  const numBlendshapes = 52
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
  for (let f = 0; f < numFrames; f++) {
    const frame = frames[f]
    const indexed = typeof frame.length === 'number'
    for (let i = 0; i < numBlendshapes; i++) {
      const raw = indexed ? frame[i] : frame[ARKIT_NAMES[i]]
      const v = raw == null ? 0 : raw
      buf[offset++] = v <= 0 ? 0 : v >= 1 ? 255 : (v * 255 + 0.5) | 0
    }
  }
  return buf
}

export { ARKIT_NAMES, encodeWAV, resampleAudio, buildAfan }
