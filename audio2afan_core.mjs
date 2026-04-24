import { LipsyncSDKNode } from '../a2f/lipsync-sdk-node.mjs'
import { buildAfan as buildAfanBuf } from './server-utils.mjs'

const ARKIT_BLENDSHAPES = LipsyncSDKNode.BLENDSHAPE_NAMES
const EXPLICIT_EMOTIONS = [
  'amazement', 'anger', 'cheekiness', 'disgust', 'fear',
  'grief', 'joy', 'outofbreath', 'pain', 'sadness'
]
const UPPER_FACE_MAX = 19
const RING_CAPACITY = 16000 * 4
const clamp = (v, lo = 0, hi = 1) => Math.max(lo, Math.min(hi, v))

function vecToBlendshapes(vec) {
  const out = new Array(ARKIT_BLENDSHAPES.length)
  for (let i = 0; i < ARKIT_BLENDSHAPES.length; i++) {
    out[i] = { name: ARKIT_BLENDSHAPES[i], value: clamp(vec[i] || 0) }
  }
  return out
}

function emptyVec() { return new Float32Array(ARKIT_BLENDSHAPES.length) }

export class Audio2FaceCore {
  static get BLENDSHAPE_NAMES() { return ARKIT_BLENDSHAPES }
  static get EMOTIONS() { return EXPLICIT_EMOTIONS }
  static toAfan(frames, fps = 30) { return buildAfanBuf(frames, fps) }

  constructor({ sampleRate, smoothingFactor, fps, langs, config } = {}) {
    this.sampleRate = sampleRate || 16000
    this.smoothingUpper = smoothingFactor ?? 0.3
    this.smoothingLower = smoothingFactor ?? 0.3
    this.fps = fps || 30
    this.bufferLen = 8320
    this.bufferOfs = 4160
    this.skinSize = 140
    this.tongueSize = 10
    this.tongueOffset = 140
    this.jawSize = 15
    this.jawOffset = 150
    this.eyesSize = 4
    this.eyesOffset = 165
    this.skinOffset = 0
    this.emotionVector = new Float32Array(26)
    this.bsWeightMultipliers = null
    this.bsWeightOffsets = null
    this.bsSolveActivePoses = null
    this.faceParams = {}
    this.lastResult = null
    this._ring = new Float32Array(RING_CAPACITY)
    this._ringLen = 0
    this._chunkLock = null
    this.session = null
    this._audioKey = null
    this._hasEmotion = false
    this._sdk = new LipsyncSDKNode({ langs: langs || ['en'], smoothing: this.smoothingUpper })
    if (config) this.loadConfig(config)
  }

  loadConfig(config) {
    const { audio_params: ap, face_params: fp, network_params: np, blendshape_params: bp } = config
    if (ap) {
      this.bufferLen = ap.buffer_len ?? this.bufferLen
      this.bufferOfs = ap.buffer_ofs ?? this.bufferOfs
      this.sampleRate = ap.samplerate ?? this.sampleRate
      if (this.bufferOfs > this.bufferLen) this.bufferOfs = this.bufferLen
    }
    if (fp) {
      this.faceParams = fp
      this.smoothingUpper = fp.upper_face_smoothing ?? this.smoothingUpper
      this.smoothingLower = fp.lower_face_smoothing ?? this.smoothingLower
      if (Array.isArray(fp.emotion)) fp.emotion.forEach((v, i) => { this.emotionVector[i] = v })
    }
    if (np) {
      this.skinSize = np.num_shapes_skin ?? this.skinSize
      this.tongueSize = np.num_shapes_tongue ?? this.tongueSize
      this.tongueOffset = this.skinSize
      this.jawOffset = this.tongueOffset + this.tongueSize
      this.jawSize = np.result_jaw_size ?? this.jawSize
      this.eyesOffset = this.jawOffset + this.jawSize
      this.eyesSize = np.result_eyes_size ?? this.eyesSize
    }
    if (bp) {
      this.bsWeightMultipliers = bp.bsWeightMultipliers ?? null
      this.bsWeightOffsets = bp.bsWeightOffsets ?? null
      this.bsSolveActivePoses = bp.bsSolveActivePoses ?? null
    }
  }

  async loadConfigFile(configPath) {
    const fs = await import('fs')
    const config = JSON.parse(fs.readFileSync(configPath, 'utf8'))
    this.loadConfig(config)
    return config
  }

  setEmotion(name, value) {
    const idx = EXPLICIT_EMOTIONS.indexOf(name)
    if (idx === -1) throw new Error(`Unknown emotion: ${name}. Valid: ${EXPLICIT_EMOTIONS.join(', ')}`)
    this.emotionVector[idx] = clamp(value)
  }
  setEmotions(obj) { for (const [k, v] of Object.entries(obj)) this.setEmotion(k, v) }
  getEmotionVector() { return new Float32Array(this.emotionVector) }

  _applyWeightMaps(vec) {
    if (!this.bsWeightMultipliers && !this.bsWeightOffsets && !this.bsSolveActivePoses) return vec
    const out = new Float32Array(vec.length)
    for (let i = 0; i < vec.length; i++) {
      let v = vec[i]
      if (this.bsWeightMultipliers) v *= this.bsWeightMultipliers[i] ?? 1
      if (this.bsWeightOffsets) v += this.bsWeightOffsets[i] ?? 0
      if (this.bsSolveActivePoses && !this.bsSolveActivePoses[i]) v = 0
      out[i] = clamp(v)
    }
    return out
  }

  // Text-driven path: produces fixed-fps Float32Array[52] frames suitable for AFAN.
  // Returns: Array<Float32Array(52)>
  processText(text, durationMs, { lang = 'en', fps } = {}) {
    const f = fps || this.fps
    const frames = this._sdk.processText(text, durationMs, { lang, fps: f })
    return frames.map(v => this._applyWeightMaps(v))
  }

  // Build AFAN binary buffer from Float32Array[52] frames.
  buildAfan(frames, fps) { return buildAfanBuf(frames, fps || this.fps) }

  // Convenience: text → AFAN buffer in one call.
  textToAfan(text, durationMs, opts = {}) {
    const fps = opts.fps || this.fps
    return this.buildAfan(this.processText(text, durationMs, { ...opts, fps }), fps)
  }

  // Audio-chunk path: kept for streaming compat. Without word timestamps we infer
  // duration from sample count and emit a neutral frame per chunk window.
  // Real lipsync arrives via processText() once the matching text is known.
  async processAudioChunk(audioData, options = {}) {
    while (this._chunkLock) await this._chunkLock
    let unlock
    this._chunkLock = new Promise(r => { unlock = r })
    try {
      if (options.emotion) this.setEmotions(options.emotion)
      if (audioData && audioData.length > 0) this._ringAppend(audioData)
      if (this._ringLen < this.bufferLen) return this.lastResult || this.getEmptyResult()
      this._ringConsume(this.bufferOfs)
      let result
      if (options.text) {
        const durMs = (this.bufferLen / this.sampleRate) * 1000
        const frames = this.processText(options.text, durMs, { lang: options.lang, fps: this.fps })
        const vec = frames[0] || emptyVec()
        result = this._frameToResult(vec)
      } else {
        result = this._frameToResult(this._applyWeightMaps(emptyVec()))
      }
      if (this.lastResult)
        result.blendshapes = this.smoothBlendshapes(this.lastResult.blendshapes, result.blendshapes)
      this.lastResult = result
      return result
    } finally { this._chunkLock = null; unlock() }
  }

  _frameToResult(vec) {
    return {
      blendshapes: vecToBlendshapes(vec),
      jaw: clamp(vec[24] || 0),
      eyes: { leftX: 0, leftY: 0, rightX: 0, rightY: 0 },
      timestamp: Date.now(),
    }
  }

  // Deprecated ONNX entry points — preserved so legacy callers fail loudly with
  // a clear message instead of mysterious undefined-method errors.
  async loadModel() { throw new Error('Audio2FaceCore: loadModel() removed. Use processText() / processAudioChunk({text}).') }
  async loadSolveData() { throw new Error('Audio2FaceCore: loadSolveData() removed. Lipsync runs without NPZ solve data.') }
  async runInference(audioData) {
    if (!this.session) throw new Error('No session loaded. Call loadModel() first.')
    return this._frameToResult(emptyVec())
  }

  _ringAppend(audioData) {
    const needed = this._ringLen + audioData.length
    if (needed > this._ring.length) {
      const newCap = Math.max(needed * 2, RING_CAPACITY)
      const grown = new Float32Array(newCap)
      grown.set(this._ring.subarray(0, this._ringLen))
      this._ring = grown
    }
    this._ring.set(audioData, this._ringLen)
    this._ringLen += audioData.length
  }
  _ringConsume(len) {
    this._ring.copyWithin(0, len, this._ringLen)
    this._ringLen -= len
  }

  smoothBlendshapes(prev, curr) {
    if (!prev || !curr || prev.length !== curr.length) return curr
    return curr.map((bs, i) => {
      const f = i <= UPPER_FACE_MAX ? this.smoothingUpper : this.smoothingLower
      return { name: bs.name, value: prev[i].value * f + bs.value * (1 - f) }
    })
  }

  aggregateResults(results) {
    if (!results.length) return this.getEmptyResult()
    const blendshapes = results[0].blendshapes.map((bs, i) => ({
      name: bs.name, value: results.reduce((a, r) => a + r.blendshapes[i].value, 0) / results.length
    }))
    return {
      blendshapes, frameCount: results.length,
      jaw: results.reduce((a, r) => a + r.jaw, 0) / results.length,
      eyes: results[results.length - 1].eyes
    }
  }

  getEmptyResult() {
    return { blendshapes: ARKIT_BLENDSHAPES.map(name => ({ name, value: 0 })), jaw: 0, eyes: null, timestamp: Date.now() }
  }

  resampleAudio(audioData, fromRate, toRate) {
    const ratio = toRate / fromRate, newLen = Math.floor(audioData.length * ratio)
    const result = new Float32Array(newLen)
    for (let i = 0; i < newLen; i++) {
      const pos = i / ratio, idx = Math.floor(pos), frac = pos - idx
      result[i] = idx >= audioData.length - 1 ? audioData[audioData.length - 1] : audioData[idx] * (1 - frac) + audioData[idx + 1] * frac
    }
    return result
  }

  setSmoothing(factor) { this.smoothingUpper = this.smoothingLower = clamp(factor) }
  setSmoothingRegion(region, factor) {
    if (region === 'upper') this.smoothingUpper = clamp(factor)
    else if (region === 'lower') this.smoothingLower = clamp(factor)
    else throw new Error(`Unknown region: ${region}. Valid: upper, lower`)
  }

  dispose() {
    this.session = null
    this._ring = new Float32Array(RING_CAPACITY)
    this._ringLen = 0
    this.lastResult = null
    this._audioKey = null
    this._hasEmotion = false
  }
}

export default Audio2FaceCore
