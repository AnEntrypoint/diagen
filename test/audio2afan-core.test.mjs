import { describe, it, expect } from 'vitest'
import { Audio2FaceCore } from '../audio2afan_core.mjs'

describe('Audio2FaceCore', () => {
  describe('constructor defaults', () => {
    const a = new Audio2FaceCore()
    it('has 52 blendshape names', () => expect(Audio2FaceCore.BLENDSHAPE_NAMES).toHaveLength(52))
    it('has 10 emotions', () => expect(Audio2FaceCore.EMOTIONS).toHaveLength(10))
    it('default sampleRate 16000', () => expect(a.sampleRate).toBe(16000))
    it('default bufferLen 8320', () => expect(a.bufferLen).toBe(8320))
    it('default bufferOfs 4160', () => expect(a.bufferOfs).toBe(4160))
    it('default smoothing 0.3', () => {
      expect(a.smoothingUpper).toBeCloseTo(0.3)
      expect(a.smoothingLower).toBeCloseTo(0.3)
    })
  })

  describe('loadConfig', () => {
    it('applies audio_params', () => {
      const a = new Audio2FaceCore()
      a.loadConfig({ audio_params: { buffer_len: 4000, buffer_ofs: 2000, samplerate: 22050 } })
      expect(a.bufferLen).toBe(4000)
      expect(a.bufferOfs).toBe(2000)
      expect(a.sampleRate).toBe(22050)
    })

    it('clamps bufferOfs to bufferLen', () => {
      const a = new Audio2FaceCore()
      a.loadConfig({ audio_params: { buffer_len: 1000, buffer_ofs: 5000 } })
      expect(a.bufferOfs).toBe(1000)
    })

    it('applies face_params', () => {
      const a = new Audio2FaceCore()
      a.loadConfig({ face_params: { upper_face_smoothing: 0.7, lower_face_smoothing: 0.1 } })
      expect(a.smoothingUpper).toBeCloseTo(0.7)
      expect(a.smoothingLower).toBeCloseTo(0.1)
    })

    it('applies network_params', () => {
      const a = new Audio2FaceCore()
      a.loadConfig({ network_params: { num_shapes_skin: 100, num_shapes_tongue: 5, result_jaw_size: 10, result_eyes_size: 2 } })
      expect(a.skinSize).toBe(100)
      expect(a.tongueSize).toBe(5)
      expect(a.tongueOffset).toBe(100)
      expect(a.jawOffset).toBe(105)
      expect(a.jawSize).toBe(10)
      expect(a.eyesOffset).toBe(115)
      expect(a.eyesSize).toBe(2)
    })

    it('applies blendshape_params', () => {
      const a = new Audio2FaceCore()
      a.loadConfig({ blendshape_params: { bsWeightMultipliers: [1, 2], bsWeightOffsets: [0.1] } })
      expect(a.bsWeightMultipliers).toEqual([1, 2])
      expect(a.bsWeightOffsets).toEqual([0.1])
    })

    it('applies emotion array from face_params', () => {
      const a = new Audio2FaceCore()
      a.loadConfig({ face_params: { emotion: [0.5, 0.3] } })
      expect(a.emotionVector[0]).toBeCloseTo(0.5)
      expect(a.emotionVector[1]).toBeCloseTo(0.3)
    })
  })

  describe('ring buffer', () => {
    it('appends and tracks length', () => {
      const a = new Audio2FaceCore()
      a._ringAppend(new Float32Array([1, 2, 3]))
      expect(a._ringLen).toBe(3)
      a._ringAppend(new Float32Array([4, 5]))
      expect(a._ringLen).toBe(5)
    })

    it('consumes from front', () => {
      const a = new Audio2FaceCore()
      a._ringAppend(new Float32Array([10, 20, 30, 40]))
      a._ringConsume(2)
      expect(a._ringLen).toBe(2)
      expect(a._ring[0]).toBe(30)
      expect(a._ring[1]).toBe(40)
    })

    it('grows when needed', () => {
      const a = new Audio2FaceCore()
      const big = new Float32Array(70000)
      a._ringAppend(big)
      expect(a._ringLen).toBe(70000)
      expect(a._ring.length).toBeGreaterThanOrEqual(70000)
    })
  })

  describe('emotions', () => {
    it('setEmotion valid', () => {
      const a = new Audio2FaceCore()
      a.setEmotion('joy', 0.9)
      expect(a.emotionVector[6]).toBeCloseTo(0.9)
    })

    it('setEmotion clamps', () => {
      const a = new Audio2FaceCore()
      a.setEmotion('joy', 2.5)
      expect(a.emotionVector[6]).toBeCloseTo(1.0)
    })

    it('setEmotion throws on invalid', () => {
      const a = new Audio2FaceCore()
      expect(() => a.setEmotion('invalid', 1)).toThrow('Unknown emotion')
    })

    it('setEmotions batch', () => {
      const a = new Audio2FaceCore()
      a.setEmotions({ joy: 0.5, anger: 0.3 })
      expect(a.emotionVector[6]).toBeCloseTo(0.5)
      expect(a.emotionVector[1]).toBeCloseTo(0.3)
    })

    it('getEmotionVector returns copy', () => {
      const a = new Audio2FaceCore()
      a.setEmotion('joy', 0.8)
      const vec = a.getEmotionVector()
      vec[6] = 0
      expect(a.emotionVector[6]).toBeCloseTo(0.8)
    })
  })

  describe('smoothBlendshapes', () => {
    it('interpolates correctly', () => {
      const a = new Audio2FaceCore()
      a.smoothingUpper = 0.3
      a.smoothingLower = 0.3
      const prev = [{ name: 'a', value: 1.0 }, { name: 'b', value: 0.0 }]
      const curr = [{ name: 'a', value: 0.0 }, { name: 'b', value: 1.0 }]
      const result = a.smoothBlendshapes(prev, curr)
      expect(result[0].value).toBeCloseTo(0.3)
      expect(result[1].value).toBeCloseTo(0.7)
    })

    it('returns curr on mismatched lengths', () => {
      const a = new Audio2FaceCore()
      const prev = [{ name: 'a', value: 1.0 }]
      const curr = [{ name: 'a', value: 0.5 }, { name: 'b', value: 0.5 }]
      expect(a.smoothBlendshapes(prev, curr)).toBe(curr)
    })

    it('returns curr if prev is null', () => {
      const a = new Audio2FaceCore()
      const curr = [{ name: 'a', value: 0.5 }]
      expect(a.smoothBlendshapes(null, curr)).toBe(curr)
    })
  })

  describe('resampleAudio', () => {
    it('downsamples', () => {
      const a = new Audio2FaceCore()
      const r = a.resampleAudio(new Float32Array([0, 0.5, 1.0, 0.5, 0]), 48000, 24000)
      expect(r.length).toBe(2)
    })

    it('upsamples', () => {
      const a = new Audio2FaceCore()
      const r = a.resampleAudio(new Float32Array([0, 1]), 8000, 16000)
      expect(r.length).toBe(4)
    })

    it('same rate returns same length', () => {
      const a = new Audio2FaceCore()
      const input = new Float32Array([0.1, 0.2, 0.3])
      const r = a.resampleAudio(input, 16000, 16000)
      expect(r.length).toBe(input.length)
    })
  })

  describe('getEmptyResult', () => {
    it('has 52 zero blendshapes', () => {
      const a = new Audio2FaceCore()
      const r = a.getEmptyResult()
      expect(r.blendshapes).toHaveLength(52)
      expect(r.blendshapes.every(b => b.value === 0)).toBe(true)
    })

    it('has jaw 0', () => expect(new Audio2FaceCore().getEmptyResult().jaw).toBe(0))
    it('has null eyes', () => expect(new Audio2FaceCore().getEmptyResult().eyes).toBeNull())
    it('has timestamp', () => expect(new Audio2FaceCore().getEmptyResult().timestamp).toBeGreaterThan(0))
  })

  describe('aggregateResults', () => {
    it('averages blendshapes', () => {
      const a = new Audio2FaceCore()
      const r = a.aggregateResults([
        { blendshapes: [{ name: 'a', value: 0.2 }], jaw: 0.3, eyes: { x: 1 } },
        { blendshapes: [{ name: 'a', value: 0.8 }], jaw: 0.7, eyes: { x: 2 } },
      ])
      expect(r.blendshapes[0].value).toBeCloseTo(0.5)
      expect(r.jaw).toBeCloseTo(0.5)
      expect(r.frameCount).toBe(2)
      expect(r.eyes).toEqual({ x: 2 })
    })

    it('empty returns emptyResult', () => {
      const a = new Audio2FaceCore()
      const r = a.aggregateResults([])
      expect(r.blendshapes).toHaveLength(52)
    })
  })

  describe('setSmoothing', () => {
    it('sets both regions', () => {
      const a = new Audio2FaceCore()
      a.setSmoothing(0.5)
      expect(a.smoothingUpper).toBeCloseTo(0.5)
      expect(a.smoothingLower).toBeCloseTo(0.5)
    })

    it('clamps to [0,1]', () => {
      const a = new Audio2FaceCore()
      a.setSmoothing(2.0)
      expect(a.smoothingUpper).toBeCloseTo(1.0)
    })
  })

  describe('setSmoothingRegion', () => {
    it('sets upper', () => {
      const a = new Audio2FaceCore()
      a.setSmoothingRegion('upper', 0.8)
      expect(a.smoothingUpper).toBeCloseTo(0.8)
    })

    it('sets lower', () => {
      const a = new Audio2FaceCore()
      a.setSmoothingRegion('lower', 0.2)
      expect(a.smoothingLower).toBeCloseTo(0.2)
    })

    it('throws on invalid region', () => {
      const a = new Audio2FaceCore()
      expect(() => a.setSmoothingRegion('middle', 0.5)).toThrow('Unknown region')
    })
  })

  describe('dispose', () => {
    it('clears all state', () => {
      const a = new Audio2FaceCore()
      a._ringLen = 100
      a.lastResult = { foo: 1 }
      a.dispose()
      expect(a.session).toBeNull()
      expect(a._ringLen).toBe(0)
      expect(a.lastResult).toBeNull()
      expect(a._audioKey).toBeNull()
    })
  })

  describe('runInference without session', () => {
    it('throws', async () => {
      const a = new Audio2FaceCore()
      await expect(a.runInference(new Float32Array(100))).rejects.toThrow('No session loaded')
    })
  })

  describe('processAudioChunk without session', () => {
    it('throws', async () => {
      const a = new Audio2FaceCore()
      await expect(a.processAudioChunk(new Float32Array(100))).rejects.toThrow('No session loaded')
    })
  })
})
