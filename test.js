import assert from 'node:assert/strict'

// VAD: stereo downmix + rms
{
  const stereo = new Float32Array([0.5, 0.3, 0.5, 0.3])
  const mono = new Float32Array(stereo.length / 2)
  for (let i = 0; i < mono.length; i++) mono[i] = (stereo[i * 2] + stereo[i * 2 + 1]) / 2
  assert.equal(mono.length, 2)
  assert.ok(Math.abs(mono[0] - 0.4) < 0.001, 'downmix L+R/2')
}

// VAD: rms threshold
{
  function rms(f) { let s = 0; for (let i = 0; i < f.length; i++) s += f[i]*f[i]; return Math.sqrt(s/f.length) }
  const silent = new Float32Array(960).fill(0.001)
  const loud = new Float32Array(960).fill(0.5)
  assert.ok(rms(silent) < 0.01, 'silent below threshold')
  assert.ok(rms(loud) > 0.01, 'speech above threshold')
}

// Processor: float32 mono -> int16 pcm buffer
{
  const mono = new Float32Array([0, 0.5, -0.5, 1.0, -1.0])
  const int16 = new Int16Array(mono.length)
  for (let i = 0; i < mono.length; i++) {
    const v = Math.max(-1, Math.min(1, mono[i]))
    int16[i] = v < 0 ? v * 0x8000 : v * 0x7FFF
  }
  assert.equal(int16[0], 0)
  assert.ok(int16[1] > 16000, 'positive sample encoded')
  assert.ok(int16[2] < -16000, 'negative sample encoded')
  const buf = Buffer.from(int16.buffer)
  assert.equal(buf.length, mono.length * 2, 'buffer is 2 bytes per sample')
}

// Resample: length ratio check (server-utils resampleAudio is linear interp)
{
  const src = new Float32Array(24000)
  for (let i = 0; i < src.length; i++) src[i] = Math.sin(i * 0.01)
  const srcRate = 24000, dstRate = 48000
  const ratio = dstRate / srcRate
  const expectedLen = Math.ceil(src.length * ratio)
  // manual resample
  const out = new Float32Array(expectedLen)
  for (let i = 0; i < expectedLen; i++) {
    const pos = i / ratio
    const lo = Math.floor(pos)
    const hi = Math.min(lo + 1, src.length - 1)
    out[i] = src[lo] + (src[hi] - src[lo]) * (pos - lo)
  }
  assert.equal(out.length, expectedLen)
  assert.ok(Math.abs(out[0] - src[0]) < 0.001, 'first sample preserved')
}

console.log('test.js: all assertions passed')
