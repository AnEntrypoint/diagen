import { describe, it, expect } from 'vitest'
import { buildTokenizer } from '../tokenizer.mjs'

const minimalTokJson = {
  model: {
    vocab: { h: 0, e: 1, l: 2, o: 3, w: 4, r: 5, d: 6, ' ': 7, he: 8, lo: 9, hel: 10, wor: 11, ld: 12, Ġ: 7 },
    merges: [['h', 'e'], ['l', 'o'], ['he', 'l'], ['w', 'o'], ['wo', 'r'], ['l', 'd']],
  },
  added_tokens: [
    { id: 100, content: '<|im_start|>' },
    { id: 101, content: '<|im_end|>' },
  ],
}

describe('buildTokenizer', () => {
  const tok = buildTokenizer(minimalTokJson)

  it('returns tokenize and decode functions', () => {
    expect(typeof tok.tokenize).toBe('function')
    expect(typeof tok.decode).toBe('function')
  })

  it('tokenizes empty string to empty array', () => {
    expect(tok.tokenize('')).toEqual([])
  })

  it('tokenizes added tokens', () => {
    const ids = tok.tokenize('<|im_start|>')
    expect(ids[0]).toBe(100)
  })

  it('recognizes added tokens at start of remaining text', () => {
    const ids = tok.tokenize('<|im_start|><|im_end|>rest')
    expect(ids[0]).toBe(100)
    expect(ids[1]).toBe(101)
  })

  it('produces integer IDs', () => {
    const ids = tok.tokenize('<|im_start|>hello')
    ids.forEach(id => expect(Number.isInteger(id)).toBe(true))
  })

  it('handles multiple added tokens in sequence', () => {
    const ids = tok.tokenize('<|im_start|><|im_end|>')
    expect(ids).toEqual([100, 101])
  })
})
