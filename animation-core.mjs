const MAGIC = 0x4146414E
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

class AnimationReader {
  constructor() {
    this.fps = 30
    this.numBlendshapes = 0
    this.numFrames = 0
    this.names = ARKIT_NAMES
    this.frames = []
  }

  fromBuffer(buf) {
    let offset = 0
    const view = new DataView(buf instanceof ArrayBuffer ? buf : buf.buffer)
    const magic = view.getUint32(offset, true); offset += 4
    if (magic !== MAGIC) throw new Error('Invalid animation file')
    const version = view.getUint8(offset); offset += 1
    if (version < 1 || version > 2) throw new Error(`Unsupported version: ${version}`)
    this.fps = view.getUint8(offset); offset += 1
    this.numBlendshapes = view.getUint8(offset); offset += 1
    offset += 1
    this.numFrames = view.getUint32(offset, true); offset += 4
    if (version === 1) {
      this.names = []
      for (let i = 0; i < this.numBlendshapes; i++) {
        const len = view.getUint8(offset++)
        this.names.push(new TextDecoder().decode(new Uint8Array(buf, offset, len)))
        offset += len
      }
    }
    this.frames = []
    for (let f = 0; f < this.numFrames; f++) {
      const frame = {}
      for (let i = 0; i < this.numBlendshapes; i++) {
        frame[this.names[i]] = view.getUint8(offset++) / 255
      }
      this.frames.push({ time: f / this.fps, blendshapes: frame })
    }
    return this
  }

  getFrameAtTime(time) {
    const index = Math.floor(time * this.fps)
    return this.frames[Math.max(0, Math.min(index, this.frames.length - 1))]
  }
}

const clamp = (v, lo = 0, hi = 1) => Math.max(lo, Math.min(hi, v))

function mapVisemes(blendshapes) {
  const {
    jawOpen = 0, mouthFunnel = 0, mouthPucker = 0,
    mouthUpperUpLeft = 0, mouthUpperUpRight = 0,
    mouthLowerDownLeft = 0, mouthLowerDownRight = 0,
    mouthStretchLeft = 0, mouthStretchRight = 0,
  } = blendshapes
  const stretch = Math.max(mouthStretchLeft, mouthStretchRight)
  const upperUp = Math.max(mouthUpperUpLeft, mouthUpperUpRight)
  const lowerDown = Math.max(mouthLowerDownLeft, mouthLowerDownRight)
  return {
    aa: clamp(jawOpen * 0.7 + lowerDown * 0.3),
    ih: clamp(upperUp * 0.6 + stretch * 0.4),
    ou: clamp(mouthFunnel * 0.5 + mouthPucker * 0.5),
    ee: clamp(stretch * 0.7 + (1 - jawOpen) * 0.3),
    oh: clamp(mouthPucker * 0.4 + jawOpen * 0.4 + mouthFunnel * 0.2),
  }
}

function mapEyes(blendshapes) {
  const {
    eyeBlinkLeft = 0, eyeBlinkRight = 0, eyeSquintLeft = 0, eyeSquintRight = 0,
    eyeWideLeft = 0, eyeWideRight = 0, eyeLookUpLeft = 0, eyeLookUpRight = 0,
    eyeLookDownLeft = 0, eyeLookDownRight = 0, eyeLookInLeft = 0, eyeLookInRight = 0,
    eyeLookOutLeft = 0, eyeLookOutRight = 0
  } = blendshapes
  return {
    blinkLeft: clamp(eyeBlinkLeft + eyeSquintLeft * 0.3),
    blinkRight: clamp(eyeBlinkRight + eyeSquintRight * 0.3),
    blink: clamp((eyeBlinkLeft + eyeBlinkRight) / 2),
    lookUp: clamp(Math.max(eyeLookUpLeft, eyeLookUpRight)),
    lookDown: clamp(Math.max(eyeLookDownLeft, eyeLookDownRight)),
    lookLeft: clamp(Math.max(eyeLookInLeft, eyeLookOutRight)),
    lookRight: clamp(Math.max(eyeLookInRight, eyeLookOutLeft))
  }
}

function mapEmotions(blendshapes) {
  const {
    mouthSmileLeft = 0, mouthSmileRight = 0, mouthFrownLeft = 0, mouthFrownRight = 0,
    browInnerUp = 0, browDownLeft = 0, browDownRight = 0, browOuterUpLeft = 0, browOuterUpRight = 0,
    cheekPuff = 0, eyeSquintLeft = 0, eyeSquintRight = 0, noseSneerLeft = 0, noseSneerRight = 0,
    jawOpen = 0, eyeWideLeft = 0, eyeWideRight = 0
  } = blendshapes
  const smile = Math.max(mouthSmileLeft, mouthSmileRight)
  const frown = Math.max(mouthFrownLeft, mouthFrownRight)
  const browDown = Math.max(browDownLeft, browDownRight)
  const squint = Math.max(eyeSquintLeft, eyeSquintRight)
  const wide = Math.max(eyeWideLeft, eyeWideRight)
  const sneer = Math.max(noseSneerLeft, noseSneerRight)
  const browUp = browInnerUp + Math.max(browOuterUpLeft, browOuterUpRight)
  return {
    happy: clamp(smile * 0.9 + squint * 0.1),
    sad: clamp(frown * 0.6 + browDown * 0.3 + (1 - smile) * 0.1),
    angry: clamp(browDown * 0.5 + sneer * 0.3 + frown * 0.2),
    relaxed: clamp((1 - browDown) * 0.5 + smile * 0.3 + cheekPuff * 0.2),
    surprised: clamp(browUp * 0.6 + wide * 0.3 + jawOpen * 0.1),
    fun: clamp(cheekPuff * 0.7 + smile * 0.3)
  }
}

export { MAGIC, ARKIT_NAMES, AnimationReader, mapVisemes, mapEyes, mapEmotions, clamp }
