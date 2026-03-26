import { describe, it, expect } from 'vitest'
import { AnimationReader, mapVisemes, mapEyes, mapEmotions, ARKIT_NAMES, MAGIC, clamp } from '../animation-core.mjs'

function buildAfanV2(frameData, fps = 30) {
  const numBS = ARKIT_NAMES.length
  const numFrames = frameData.length
  const buf = new ArrayBuffer(12 + numFrames * numBS)
  const view = new DataView(buf)
  view.setUint32(0, MAGIC, true)
  view.setUint8(4, 2)
  view.setUint8(5, fps)
  view.setUint8(6, numBS)
  view.setUint8(7, 0)
  view.setUint32(8, numFrames, true)
  let offset = 12
  for (const frame of frameData) {
    for (let i = 0; i < numBS; i++) {
      view.setUint8(offset++, Math.round((frame[ARKIT_NAMES[i]] || 0) * 255))
    }
  }
  return buf
}

describe('AnimationReader', () => {
  it('parses v2 buffer', () => {
    const frame = {}
    ARKIT_NAMES.forEach(n => { frame[n] = 0.75 })
    const buf = buildAfanV2([frame, frame])
    const reader = new AnimationReader().fromBuffer(buf)
    expect(reader.fps).toBe(30)
    expect(reader.numFrames).toBe(2)
    expect(reader.numBlendshapes).toBe(52)
    expect(reader.frames).toHaveLength(2)
  })

  it('decodes values correctly', () => {
    const frame = {}
    ARKIT_NAMES.forEach(n => { frame[n] = 0.5 })
    const reader = new AnimationReader().fromBuffer(buildAfanV2([frame]))
    expect(reader.frames[0].blendshapes.jawOpen).toBeCloseTo(128 / 255, 2)
  })

  it('assigns correct timestamps', () => {
    const frame = {}
    ARKIT_NAMES.forEach(n => { frame[n] = 0 })
    const reader = new AnimationReader().fromBuffer(buildAfanV2([frame, frame, frame], 30))
    expect(reader.frames[0].time).toBeCloseTo(0)
    expect(reader.frames[1].time).toBeCloseTo(1 / 30, 4)
    expect(reader.frames[2].time).toBeCloseTo(2 / 30, 4)
  })

  it('throws on invalid magic', () => {
    const buf = new ArrayBuffer(12)
    expect(() => new AnimationReader().fromBuffer(buf)).toThrow('Invalid animation file')
  })

  it('throws on unsupported version', () => {
    const buf = new ArrayBuffer(12)
    const view = new DataView(buf)
    view.setUint32(0, MAGIC, true)
    view.setUint8(4, 5)
    expect(() => new AnimationReader().fromBuffer(buf)).toThrow('Unsupported version')
  })

  it('getFrameAtTime returns correct frame', () => {
    const frame = {}
    ARKIT_NAMES.forEach(n => { frame[n] = 0.5 })
    const reader = new AnimationReader().fromBuffer(buildAfanV2([frame, frame, frame], 30))
    const f = reader.getFrameAtTime(1 / 30)
    expect(f).toBeDefined()
    expect(f.time).toBeCloseTo(1 / 30, 4)
  })

  it('getFrameAtTime clamps to bounds', () => {
    const frame = {}
    ARKIT_NAMES.forEach(n => { frame[n] = 0 })
    const reader = new AnimationReader().fromBuffer(buildAfanV2([frame], 30))
    expect(reader.getFrameAtTime(-1)).toBe(reader.frames[0])
    expect(reader.getFrameAtTime(999)).toBe(reader.frames[0])
  })

  it('handles zero frames', () => {
    const buf = new ArrayBuffer(12)
    const view = new DataView(buf)
    view.setUint32(0, MAGIC, true)
    view.setUint8(4, 2)
    view.setUint8(5, 30)
    view.setUint8(6, 52)
    view.setUint32(8, 0, true)
    const reader = new AnimationReader().fromBuffer(buf)
    expect(reader.numFrames).toBe(0)
    expect(reader.frames).toHaveLength(0)
  })
})

describe('mapVisemes', () => {
  it('returns all 5 keys', () => {
    expect(Object.keys(mapVisemes({}))).toEqual(['aa', 'ih', 'ou', 'ee', 'oh'])
  })

  it('all values in [0,1]', () => {
    const v = mapVisemes({ jawOpen: 2, mouthFunnel: -1, mouthPucker: 5 })
    Object.values(v).forEach(val => {
      expect(val).toBeGreaterThanOrEqual(0)
      expect(val).toBeLessThanOrEqual(1)
    })
  })

  it('jawOpen drives aa', () => {
    const v = mapVisemes({ jawOpen: 1.0 })
    expect(v.aa).toBeGreaterThan(0)
  })

  it('empty blendshapes give ee > 0 due to (1-jawOpen)*0.3', () => {
    const v = mapVisemes({})
    expect(v.ee).toBeCloseTo(0.3)
  })
})

describe('mapEyes', () => {
  it('returns all 7 keys', () => {
    const keys = Object.keys(mapEyes({}))
    expect(keys).toEqual(['blinkLeft', 'blinkRight', 'blink', 'lookUp', 'lookDown', 'lookLeft', 'lookRight'])
  })

  it('blink averages left and right', () => {
    const e = mapEyes({ eyeBlinkLeft: 0.8, eyeBlinkRight: 0.4 })
    expect(e.blink).toBeCloseTo(0.6)
  })

  it('clamps overflow', () => {
    const e = mapEyes({ eyeBlinkLeft: 1.0, eyeSquintLeft: 1.0 })
    expect(e.blinkLeft).toBeLessThanOrEqual(1.0)
  })
})

describe('mapEmotions', () => {
  it('returns all 6 keys', () => {
    const keys = Object.keys(mapEmotions({}))
    expect(keys).toEqual(['happy', 'sad', 'angry', 'relaxed', 'surprised', 'fun'])
  })

  it('smile drives happy', () => {
    const e = mapEmotions({ mouthSmileLeft: 1.0, mouthSmileRight: 1.0 })
    expect(e.happy).toBeGreaterThan(0.5)
  })

  it('all values in [0,1]', () => {
    const e = mapEmotions({ browInnerUp: 5, mouthSmileLeft: -1, cheekPuff: 3 })
    Object.values(e).forEach(val => {
      expect(val).toBeGreaterThanOrEqual(0)
      expect(val).toBeLessThanOrEqual(1)
    })
  })

  it('relaxed is high with neutral face', () => {
    const e = mapEmotions({})
    expect(e.relaxed).toBeGreaterThan(0.3)
  })
})

describe('clamp', () => {
  it('clamps above max', () => expect(clamp(1.5)).toBe(1))
  it('clamps below min', () => expect(clamp(-0.5)).toBe(0))
  it('preserves in-range', () => expect(clamp(0.5)).toBe(0.5))
  it('custom range', () => expect(clamp(5, 0, 10)).toBe(5))
})
