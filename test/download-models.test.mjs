import { describe, it, expect } from 'vitest'
import fs from 'fs'
import path from 'path'

describe('download-models structure', () => {
  it('module exports downloadModels function', async () => {
    const mod = await import('../download-models.js')
    expect(typeof mod.downloadModels).toBe('function')
  })

  it('IPFS gateways are valid URLs', async () => {
    const src = fs.readFileSync(path.join(process.cwd(), 'download-models.js'), 'utf8')
    const urlMatches = src.match(/https:\/\/[^']+/g)
    expect(urlMatches.length).toBeGreaterThanOrEqual(4)
    urlMatches.forEach(url => {
      expect(url).toMatch(/^https:\/\//)
      expect(url).toContain('ipfs')
    })
  })

  it('model configs have required fields', async () => {
    const src = fs.readFileSync(path.join(process.cwd(), 'download-models.js'), 'utf8')
    expect(src).toContain("cid: '")
    expect(src).toContain("dir: ")
    expect(src).toContain("files: [")
  })

  it('audio2afan model includes config.json and model.onnx', async () => {
    const src = fs.readFileSync(path.join(process.cwd(), 'download-models.js'), 'utf8')
    expect(src).toContain("'config.json'")
    expect(src).toContain("'model.onnx'")
    expect(src).toContain("'solve_data.npz'")
  })

  it('tts model includes required ONNX files', async () => {
    const src = fs.readFileSync(path.join(process.cwd(), 'download-models.js'), 'utf8')
    expect(src).toContain('flow_lm_flow_int8.onnx')
    expect(src).toContain('mimi_decoder_int8.onnx')
    expect(src).toContain('text_conditioner.onnx')
  })
})
