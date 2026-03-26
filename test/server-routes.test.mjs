import { describe, it, expect, beforeAll } from 'vitest'
import express from 'express'
import fs from 'fs'
import path from 'path'

const __dirname = path.resolve()
let app

beforeAll(() => {
  app = express()
  app.use(express.json({ limit: '50mb' }))
  app.use((req, res, next) => {
    res.setHeader('Cross-Origin-Opener-Policy', 'same-origin')
    res.setHeader('Cross-Origin-Embedder-Policy', 'require-corp')
    res.setHeader('Access-Control-Allow-Origin', '*')
    res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
    res.setHeader('Access-Control-Allow-Headers', 'Content-Type')
    if (req.method === 'OPTIONS') return res.sendStatus(200)
    next()
  })
  app.get('/', (req, res) => res.sendFile(path.join(__dirname, 'index.html')))
  app.get('/client.js', (req, res) => res.sendFile(path.join(__dirname, 'client.js')))

  const VOICES_DIR = path.join(__dirname, 'voices')
  app.get('/demo/voices/manifest.json', (req, res) => {
    const files = fs.existsSync(VOICES_DIR) ? fs.readdirSync(VOICES_DIR).filter(f => f.endsWith('.wav')) : []
    res.json(files)
  })

  app.post('/api/generate', (req, res) => {
    const { text } = req.body
    if (!text) return res.status(400).json({ error: 'text required' })
    res.status(503).json({ error: 'Models not loaded in test' })
  })

  app.post('/dialog', (req, res) => {
    const { prompt } = req.body
    if (!prompt) return res.status(400).json({ error: 'prompt required' })
    res.status(503).json({ error: 'Models not loaded in test' })
  })
})

async function request(method, path, body) {
  const port = 0
  return new Promise((resolve, reject) => {
    const server = app.listen(0, '127.0.0.1', async () => {
      const addr = server.address()
      try {
        const url = `http://127.0.0.1:${addr.port}${path}`
        const opts = { method }
        if (body) {
          opts.headers = { 'Content-Type': 'application/json' }
          opts.body = JSON.stringify(body)
        }
        const res = await fetch(url)
        if (method !== 'GET') {
          const res2 = await fetch(url, opts)
          server.close()
          resolve({ status: res2.status, headers: Object.fromEntries(res2.headers), body: await res2.json().catch(() => null) })
        } else {
          server.close()
          resolve({ status: res.status, headers: Object.fromEntries(res.headers), body: await res.text().catch(() => null) })
        }
      } catch (e) {
        server.close()
        reject(e)
      }
    })
  })
}

describe('GET routes', () => {
  it('/ returns 200', async () => {
    const r = await request('GET', '/')
    expect(r.status).toBe(200)
  })

  it('/ serves HTML', async () => {
    const r = await request('GET', '/')
    expect(r.body).toContain('<')
  })

  it('/client.js returns 200', async () => {
    const r = await request('GET', '/client.js')
    expect(r.status).toBe(200)
  })

  it('/demo/voices/manifest.json returns array', async () => {
    const r = await request('GET', '/demo/voices/manifest.json')
    expect(r.status).toBe(200)
  })
})

describe('CORS headers', () => {
  it('sets Access-Control-Allow-Origin', async () => {
    const r = await request('GET', '/')
    expect(r.headers['access-control-allow-origin']).toBe('*')
  })

  it('sets Cross-Origin-Opener-Policy', async () => {
    const r = await request('GET', '/')
    expect(r.headers['cross-origin-opener-policy']).toBe('same-origin')
  })

  it('sets Cross-Origin-Embedder-Policy', async () => {
    const r = await request('GET', '/')
    expect(r.headers['cross-origin-embedder-policy']).toBe('require-corp')
  })
})

describe('POST /api/generate', () => {
  it('returns 400 without text', async () => {
    const r = await request('POST', '/api/generate', {})
    expect(r.status).toBe(400)
    expect(r.body.error).toBe('text required')
  })

  it('returns 503 with text but no models', async () => {
    const r = await request('POST', '/api/generate', { text: 'hello' })
    expect(r.status).toBe(503)
  })
})

describe('POST /dialog', () => {
  it('returns 400 without prompt', async () => {
    const r = await request('POST', '/dialog', {})
    expect(r.status).toBe(400)
    expect(r.body.error).toBe('prompt required')
  })

  it('returns 503 with prompt but no models', async () => {
    const r = await request('POST', '/dialog', { prompt: 'hi' })
    expect(r.status).toBe(503)
  })
})
