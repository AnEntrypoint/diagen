import { describe, it, expect } from 'vitest'
import fs from 'fs'
import path from 'path'

describe('download-models structure', () => {
  it('module exports downloadModels function', async () => {
    const mod = await import('../download-models.js')
    expect(typeof mod.downloadModels).toBe('function')
  })

  it('verifies local models under models/ (LFS-backed)', () => {
    const src = fs.readFileSync(path.join(process.cwd(), 'download-models.js'), 'utf8')
    expect(src).toContain("'models'")
    expect(src).toContain('git lfs pull')
    expect(src).not.toContain('ipfs')
  })

  it('tts model includes required ONNX files', () => {
    const src = fs.readFileSync(path.join(process.cwd(), 'download-models.js'), 'utf8')
    expect(src).toContain('flow_lm_flow_int8.onnx')
    expect(src).toContain('mimi_decoder_int8.onnx')
    expect(src).toContain('text_conditioner.onnx')
  })
})
