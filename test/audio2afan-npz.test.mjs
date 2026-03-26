import { describe, it, expect } from 'vitest'
import { parseNpy } from '../audio2afan_npz.mjs'

function buildNpyBuffer(data, shape, dtype = '<f4', version = 1) {
  const header = `{'descr': '${dtype}', 'fortran_order': False, 'shape': (${shape.join(', ')}${shape.length === 1 ? ',' : ''}), }`
  const magicLen = 6 + 2 + (version >= 2 ? 4 : 2)
  const padLen = 64 - ((magicLen + header.length) % 64)
  const paddedHeader = header + ' '.repeat(padLen - 1) + '\n'
  const headerBytes = new TextEncoder().encode(paddedHeader)

  let typedData
  if (dtype === '<f4') typedData = new Uint8Array(new Float32Array(data).buffer)
  else if (dtype === '<f8') typedData = new Uint8Array(new Float64Array(data).buffer)
  else if (dtype === '<i4') typedData = new Uint8Array(new Int32Array(data).buffer)
  else if (dtype === '|u1') typedData = new Uint8Array(data)
  else typedData = new Uint8Array(new Float32Array(data).buffer)

  const magic = new Uint8Array([0x93, 0x4E, 0x55, 0x4D, 0x50, 0x59, version, 0])
  let headerLenBuf
  if (version >= 2) {
    headerLenBuf = new Uint8Array(4)
    new DataView(headerLenBuf.buffer).setUint32(0, headerBytes.length, true)
  } else {
    headerLenBuf = new Uint8Array(2)
    new DataView(headerLenBuf.buffer).setUint16(0, headerBytes.length, true)
  }

  const result = new Uint8Array(magic.length + headerLenBuf.length + headerBytes.length + typedData.length)
  let offset = 0
  result.set(magic, offset); offset += magic.length
  result.set(headerLenBuf, offset); offset += headerLenBuf.length
  result.set(headerBytes, offset); offset += headerBytes.length
  result.set(typedData, offset)
  return result
}

describe('parseNpy', () => {
  it('parses float32 2D array', () => {
    const npy = buildNpyBuffer([1, 2, 3, 4], [2, 2], '<f4')
    const { data, shape } = parseNpy(npy)
    expect(shape).toEqual([2, 2])
    expect(Array.from(data)).toEqual([1, 2, 3, 4])
    expect(data.constructor.name).toBe('Float32Array')
  })

  it('parses float64', () => {
    const npy = buildNpyBuffer([1.5, 2.5], [2], '<f8')
    const { data, shape } = parseNpy(npy)
    expect(shape).toEqual([2])
    expect(data[0]).toBeCloseTo(1.5)
    expect(data.constructor.name).toBe('Float64Array')
  })

  it('parses int32', () => {
    const npy = buildNpyBuffer([10, 20, 30], [3], '<i4')
    const { data, shape } = parseNpy(npy)
    expect(shape).toEqual([3])
    expect(Array.from(data)).toEqual([10, 20, 30])
    expect(data.constructor.name).toBe('Int32Array')
  })

  it('parses uint8', () => {
    const npy = buildNpyBuffer([0, 128, 255], [3], '|u1')
    const { data, shape } = parseNpy(npy)
    expect(shape).toEqual([3])
    expect(Array.from(data)).toEqual([0, 128, 255])
    expect(data.constructor.name).toBe('Uint8Array')
  })

  it('parses NPY v2 header', () => {
    const npy = buildNpyBuffer([1, 2], [2], '<f4', 2)
    const { data, shape } = parseNpy(npy)
    expect(shape).toEqual([2])
    expect(Array.from(data)).toEqual([1, 2])
  })

  it('parses 1D shape with trailing comma', () => {
    const npy = buildNpyBuffer([5], [1], '<f4')
    const { data, shape } = parseNpy(npy)
    expect(shape).toEqual([1])
  })

  it('parses empty shape', () => {
    const npy = buildNpyBuffer([], [], '<f4')
    const { data, shape } = parseNpy(npy)
    expect(shape).toEqual([])
    expect(data.length).toBe(0)
  })

  it('throws on invalid header', () => {
    const bad = new Uint8Array([0x93, 0x4E, 0x55, 0x4D, 0x50, 0x59, 1, 0, 10, 0])
    const header = new TextEncoder().encode("{'descr': 'invalid', 'shape': (2,), }" + ' '.repeat(10) + '\n')
    const buf = new Uint8Array(bad.length + header.length)
    buf.set(bad); buf.set(header, bad.length)
    expect(() => parseNpy(buf)).toThrow('Cannot parse numpy dtype')
  })
})
