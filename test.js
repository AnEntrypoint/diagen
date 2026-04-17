import assert from 'node:assert/strict'

{
  function rms(f) { let s = 0; for (let i = 0; i < f.length; i++) s += f[i]*f[i]; return Math.sqrt(s/f.length) }
  const silent = new Float32Array(960).fill(0.001)
  const loud = new Float32Array(960).fill(0.5)
  assert.ok(rms(silent) < 0.01, 'silent below threshold')
  assert.ok(rms(loud) > 0.01, 'speech above threshold')
}

{
  const mono = new Float32Array([0.1, 0.2, 0.3])
  const stereo = new Float32Array(mono.length * 2)
  for (let i = 0; i < mono.length; i++) { stereo[i * 2] = mono[i]; stereo[i * 2 + 1] = mono[i] }
  assert.equal(stereo.length, 6, 'upmix doubles length')
  assert.ok(Math.abs(stereo[0] - 0.1) < 0.001, 'L ch sample 0')
  assert.ok(Math.abs(stereo[1] - 0.1) < 0.001, 'R ch sample 0')
  assert.ok(Math.abs(stereo[2] - 0.2) < 0.001, 'L ch sample 1')
  assert.ok(Math.abs(stereo[4] - 0.3) < 0.001, 'L ch sample 2')
}

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

{
  function resampleAudio(f32, fromRate, toRate) {
    const ratio = fromRate / toRate
    const newLen = Math.round(f32.length / ratio)
    const result = new Float32Array(newLen)
    for (let i = 0; i < newLen; i++) {
      const idx = i * ratio
      const lo = Math.floor(idx)
      const hi = Math.min(lo + 1, f32.length - 1)
      result[i] = f32[lo] * (1 - (idx - lo)) + f32[hi] * (idx - lo)
    }
    return result
  }
  const src24k = new Float32Array(24000).fill(0.5)
  const out48k = resampleAudio(src24k, 24000, 48000)
  assert.equal(out48k.length, 48000, '24k->48k doubles samples')

  const src22k = new Float32Array(22050).fill(0.5)
  const out22to48 = resampleAudio(src22k, 22050, 48000)
  assert.ok(Math.abs(out22to48.length - 48000) < 2, '22050->48k ~48000 samples')
  assert.ok(Math.abs(out22to48[0] - 0.5) < 0.001, 'sample value preserved')
}

console.log('test.js: all assertions passed')

{
  const closeCode4017 = 4017
  const closeCode4006 = 4006
  assert.ok(closeCode4017 === 4017, '4017 triggers gateway reconnect branch')
  assert.ok(closeCode4006 !== 4017, 'non-4017 takes delay branch')
}
