import { describe, it, expect } from 'vitest'
import { ARKIT_NAMES, encodeWAV, resampleAudio, buildAfan } from '../server-utils.mjs'

describe('ARKIT_NAMES', () => {
  it('has 52 entries', () => expect(ARKIT_NAMES).toHaveLength(52))
  it('includes jawOpen', () => expect(ARKIT_NAMES).toContain('jawOpen'))
  it('all unique', () => expect(new Set(ARKIT_NAMES).size).toBe(52))
})

describe('encodeWAV', () => {
  it('produces valid RIFF header', () => {
    const wav = encodeWAV(new Float32Array([0, 0.5, -0.5]), 24000)
    expect(String.fromCharCode(wav[0], wav[1], wav[2], wav[3])).toBe('RIFF')
    expect(String.fromCharCode(wav[8], wav[9], wav[10], wav[11])).toBe('WAVE')
  })

  it('encodes correct sample rate', () => {
    const wav = encodeWAV(new Float32Array([0]), 44100)
    const view = new DataView(wav.buffer, wav.byteOffset, wav.byteLength)
    expect(view.getUint32(24, true)).toBe(44100)
  })

  it('encodes correct data size', () => {
    const samples = new Float32Array([0, 0.5, -0.5, 1.0])
    const wav = encodeWAV(samples, 24000)
    const view = new DataView(wav.buffer, wav.byteOffset, wav.byteLength)
    expect(view.getUint32(40, true)).toBe(samples.length * 2)
  })

  it('total length = 44 + samples*2', () => {
    const samples = new Float32Array(100)
    const wav = encodeWAV(samples, 16000)
    expect(wav.length).toBe(44 + 100 * 2)
  })

  it('clamps values to [-1, 1]', () => {
    const wav = encodeWAV(new Float32Array([5.0, -5.0]), 16000)
    const view = new DataView(wav.buffer, wav.byteOffset, wav.byteLength)
    expect(view.getInt16(44, true)).toBe(0x7FFF)
    expect(view.getInt16(46, true)).toBe(-0x8000)
  })

  it('handles empty input', () => {
    const wav = encodeWAV(new Float32Array(0), 24000)
    expect(wav.length).toBe(44)
  })

  it('PCM format = 1', () => {
    const wav = encodeWAV(new Float32Array([0]), 16000)
    const view = new DataView(wav.buffer, wav.byteOffset, wav.byteLength)
    expect(view.getUint16(20, true)).toBe(1)
  })

  it('mono channel', () => {
    const wav = encodeWAV(new Float32Array([0]), 16000)
    const view = new DataView(wav.buffer, wav.byteOffset, wav.byteLength)
    expect(view.getUint16(22, true)).toBe(1)
  })

  it('16-bit depth', () => {
    const wav = encodeWAV(new Float32Array([0]), 16000)
    const view = new DataView(wav.buffer, wav.byteOffset, wav.byteLength)
    expect(view.getUint16(34, true)).toBe(16)
  })
})

describe('resampleAudio', () => {
  it('downsamples 2:1', () => {
    const input = new Float32Array([0, 0.25, 0.5, 0.75, 1.0])
    const r = resampleAudio(input, 48000, 24000)
    expect(r.length).toBe(3)
  })

  it('same rate preserves length', () => {
    const input = new Float32Array([0.1, 0.2, 0.3])
    const r = resampleAudio(input, 16000, 16000)
    expect(r.length).toBe(3)
  })

  it('preserves first sample', () => {
    const r = resampleAudio(new Float32Array([0.5, 0.8, 0.2]), 16000, 8000)
    expect(r[0]).toBeCloseTo(0.5)
  })

  it('handles single sample', () => {
    const r = resampleAudio(new Float32Array([0.7]), 48000, 24000)
    expect(r.length).toBeGreaterThanOrEqual(1)
  })

  it('upsamples', () => {
    const input = new Float32Array([0, 1])
    const r = resampleAudio(input, 8000, 24000)
    expect(r.length).toBe(6)
  })

  it('interpolates linearly', () => {
    const r = resampleAudio(new Float32Array([0, 1, 0]), 24000, 24000)
    expect(r[0]).toBeCloseTo(0)
    expect(r[1]).toBeCloseTo(1)
    expect(r[2]).toBeCloseTo(0)
  })
})

describe('buildAfan', () => {
  it('writes correct magic', () => {
    const frame = {}
    ARKIT_NAMES.forEach(n => { frame[n] = 0 })
    const buf = buildAfan([frame])
    expect(buf.readUInt32LE(0)).toBe(0x4146414E)
  })

  it('writes version 2', () => {
    const frame = {}
    ARKIT_NAMES.forEach(n => { frame[n] = 0 })
    expect(buildAfan([frame])[4]).toBe(2)
  })

  it('writes correct fps', () => {
    const frame = {}
    ARKIT_NAMES.forEach(n => { frame[n] = 0 })
    expect(buildAfan([frame], 60)[5]).toBe(60)
  })

  it('writes 52 blendshapes', () => {
    const frame = {}
    ARKIT_NAMES.forEach(n => { frame[n] = 0 })
    expect(buildAfan([frame])[6]).toBe(52)
  })

  it('writes correct frame count', () => {
    const frame = {}
    ARKIT_NAMES.forEach(n => { frame[n] = 0 })
    const buf = buildAfan([frame, frame, frame])
    expect(buf.readUInt32LE(8)).toBe(3)
  })

  it('encodes 0.5 as 128', () => {
    const frame = {}
    ARKIT_NAMES.forEach(n => { frame[n] = 0.5 })
    const buf = buildAfan([frame])
    expect(buf[12]).toBe(128)
  })

  it('clamps values to [0, 255]', () => {
    const frame = {}
    ARKIT_NAMES.forEach(n => { frame[n] = 2.0 })
    const buf = buildAfan([frame])
    expect(buf[12]).toBe(255)

    const frame2 = {}
    ARKIT_NAMES.forEach(n => { frame2[n] = -1.0 })
    const buf2 = buildAfan([frame2])
    expect(buf2[12]).toBe(0)
  })

  it('total size = 12 + frames * 52', () => {
    const frame = {}
    ARKIT_NAMES.forEach(n => { frame[n] = 0 })
    const buf = buildAfan([frame, frame])
    expect(buf.length).toBe(12 + 2 * 52)
  })

  it('handles missing blendshape keys as 0', () => {
    const buf = buildAfan([{}])
    expect(buf[12]).toBe(0)
  })
})
