import * as THREE from 'three'
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js'
import { VRMLoaderPlugin, VRMUtils } from '@pixiv/three-vrm'
import { OrbitControls } from 'three/addons/controls/OrbitControls.js'
import { AnimationReader, mapVisemes, mapEyes } from './animation-core.mjs'

class IdleAnimator {
  constructor(vrm) {
    this.vrm = vrm
    this.time = 0
    this.blinkState = { isBlinking: false, blinkTimer: 0, nextBlink: Math.random() * 2 + 2 }
    this.breathingPhase = Math.random() * Math.PI * 2
    this.microMovements = {
      browPhase: Math.random() * Math.PI * 2,
      mouthPhase: Math.random() * Math.PI * 2,
      lookPhase: Math.random() * Math.PI * 2
    }
  }

  update(deltaTime) {
    this.time += deltaTime

    const expressions = new Map()
    const clamp = (v, lo = 0, hi = 1) => Math.max(lo, Math.min(hi, v))

    this.breathingPhase += deltaTime * 1.5
    const breathValue = clamp((Math.sin(this.breathingPhase) + 1) * 0.5 * 0.15)
    expressions.set('neutral', 1 - breathValue)

    this.microMovements.browPhase += deltaTime * 0.3
    const browSubtle = clamp((Math.sin(this.microMovements.browPhase) + 1) * 0.5 * 0.08)
    if (Math.random() > 0.995) this.microMovements.browPhase += Math.random() * 0.5

    this.blinkState.blinkTimer += deltaTime
    if (!this.blinkState.isBlinking && this.blinkState.blinkTimer >= this.blinkState.nextBlink) {
      this.blinkState.isBlinking = true
      this.blinkState.blinkTimer = 0
    }
    if (this.blinkState.isBlinking) {
      const blinkProgress = this.blinkState.blinkTimer / 0.15
      if (blinkProgress >= 1) {
        this.blinkState.isBlinking = false
        this.blinkState.blinkTimer = 0
        this.blinkState.nextBlink = Math.random() * 3 + 2
      } else {
        const blinkValue = Math.sin(blinkProgress * Math.PI)
        expressions.set('blink', clamp(blinkValue))
        expressions.set('blinkLeft', clamp(blinkValue))
        expressions.set('blinkRight', clamp(blinkValue))
      }
    }

    const hasExpression = (name) => {
      if (this.vrm.expressionManager) return this.vrm.expressionManager.expressions.some(e => e.expressionName === name)
      if (this.vrm.blendShapeProxy) return this.vrm.blendShapeProxy.getExpressionNames().includes(name)
      return false
    }

    return expressions
  }
}

class FacialAnimationPlayer {
  constructor(vrm) {
    this.vrm = vrm
    this.animation = null
    this.audio = null
    this.isPlaying = false
    this.startTime = 0
    this.availableExpressions = new Set()
    this.lastApplied = new Map()
    this.morphTargetMeshes = []
    this.idleAnimator = new IdleAnimator(vrm)
    this.currentExpressions = new Map()

    if (vrm.blendShapeProxy) {
      this.vrmVersion = '0.0'
      this.blendProxy = vrm.blendShapeProxy
      const presets = ['A','I','U','E','O','Blink','Blink_L','Blink_R','Neutral','LookUp','LookDown','LookLeft','LookRight']
      presets.forEach(p => this.availableExpressions.add(p))
    } else if (vrm.expressionManager) {
      this.vrmVersion = '1.0'
      this.expressionManager = vrm.expressionManager
      vrm.expressionManager.expressions.forEach(e => this.availableExpressions.add(e.expressionName))
    } else {
      this.vrmVersion = 'arkit'
      vrm.scene.traverse(obj => {
        if (obj.isMesh && obj.morphTargetDictionary) this.morphTargetMeshes.push(obj)
      })
    }

    console.log(`[vrm] Detected VRM ${this.vrmVersion}, expressions:`, [...this.availableExpressions].join(', ') || 'morph targets')
  }

  loadAnimation(buffer) {
    this.animation = new AnimationReader().fromBuffer(buffer)
    return this.animation
  }

  loadAudio(base64) {
    this.audio = new Audio()
    this.audio.src = `data:audio/wav;base64,${base64}`
    return new Promise((resolve, reject) => {
      this.audio.oncanplaythrough = () => resolve(this)
      this.audio.onerror = () => reject(new Error('Failed to load audio'))
    })
  }

  async load(animBase64, audioBase64) {
    if (animBase64) {
      const animBuffer = Uint8Array.from(atob(animBase64), c => c.charCodeAt(0)).buffer
      this.loadAnimation(animBuffer)
    }
    if (audioBase64) {
      await this.loadAudio(audioBase64)
    }
    return this
  }

  play() {
    if (!this.animation) return
    this.isPlaying = true
    this.startTime = performance.now()
    this.lastApplied.clear()
    
    if (this.audio) {
      this.audio.currentTime = 0
      this.audio.play().catch(() => {})
    }
  }

  stop() {
    this.isPlaying = false
    if (this.audio) {
      this.audio.pause()
      this.audio.currentTime = 0
    }
    this.resetExpressions()
  }

  resetExpressions() {
    this.currentExpressions.clear()
    if (this.vrmVersion === '0.0') {
      for (const name of this.availableExpressions) this.blendProxy.setValue(name, 0)
    } else if (this.vrmVersion === '1.0') {
      for (const name of this.availableExpressions) this.expressionManager.setValue(name, 0)
    } else {
      for (const mesh of this.morphTargetMeshes) {
        if (mesh.morphTargetInfluences) mesh.morphTargetInfluences.fill(0)
      }
    }
    this.lastApplied.clear()
  }

  update(deltaTime = 1/60) {
    if (this.isPlaying && this.animation) {
      const elapsed = (performance.now() - this.startTime) / 1000
      const frame = this.animation.getFrameAtTime(elapsed)
      if (frame) {
        this.applyFrame(frame.blendshapes)
      }
      if (elapsed >= this.animation.frames.length / this.animation.fps) {
        this.isPlaying = false
        this.resetExpressions()
      }
    } else {
      const idleExpressions = this.idleAnimator.update(deltaTime)
      this._applyIdleExpressions(idleExpressions)
    }
  }

  _applyIdleExpressions(idleExpressions) {
    const values = new Map()
    const has = (n) => this.availableExpressions.has(n)

    if (this.vrmVersion === '1.0') {
      for (const [name, val] of idleExpressions) {
        if (has(name)) values.set(name, val)
      }
      for (const [name, val] of values) {
        this.expressionManager.setValue(name, val)
      }
      for (const name of this.lastApplied.keys()) {
        if (!values.has(name)) this.expressionManager.setValue(name, 0)
      }
      this.lastApplied = new Map(values)
    } else if (this.vrmVersion === '0.0') {
      for (const [name, val] of idleExpressions) {
        const vrm0Name = name === 'blink' ? 'Blink' : name === 'blinkLeft' ? 'Blink_L' : name === 'blinkRight' ? 'Blink_R' : name === 'neutral' ? 'Neutral' : name
        if (has(vrm0Name)) this.blendProxy.setValue(vrm0Name, val)
      }
    }
  }

  applyFrame(blendshapes) {
    if (this.vrmVersion === '0.0') return this._applyVrm0(blendshapes)
    if (this.vrmVersion === '1.0') return this._applyVrm1(blendshapes)
    this._applyArkit(blendshapes)
  }

  _dominantViseme(visemes) {
    return Object.entries(visemes).reduce((a, b) => b[1] > a[1] ? b : a, ['', 0])
  }

  _applyVrm0(blendshapes) {
    const clamp = (v, lo = 0, hi = 1) => Math.max(lo, Math.min(hi, v))
    const has = (n) => this.availableExpressions.has(n)
    const set = (n, v) => { if (has(n)) this.blendProxy.setValue(n, clamp(v)) }

    const vrm0Visemes = { A: 0, I: 0, U: 0, E: 0, O: 0 }
    const v = mapVisemes(blendshapes)
    vrm0Visemes.A = v.aa; vrm0Visemes.I = v.ih; vrm0Visemes.U = v.ou; vrm0Visemes.E = v.ee; vrm0Visemes.O = v.oh
    const [dom, domVal] = this._dominantViseme(vrm0Visemes)
    for (const k of ['A','I','U','E','O']) set(k, k === dom ? domVal : 0)

    const eyes = mapEyes(blendshapes)
    set('Blink_L', eyes.blinkLeft * 0.8)
    set('Blink_R', eyes.blinkRight * 0.8)
    set('Blink', eyes.blink * 0.8)
  }

  _applyVrm1(blendshapes) {
    const clamp = (v, lo = 0, hi = 1) => Math.max(lo, Math.min(hi, v))
    const values = new Map()
    const has = (n) => this.availableExpressions.has(n)
    const set = (n, v) => { if (has(n)) values.set(n, clamp(v)) }

    const v = mapVisemes(blendshapes)
    const [dom, domVal] = this._dominantViseme(v)
    for (const k of ['aa','ih','ou','ee','oh']) set(k, k === dom ? domVal : 0)

    const eyes = mapEyes(blendshapes)
    set('blinkLeft', eyes.blinkLeft * 0.8)
    set('blinkRight', eyes.blinkRight * 0.8)
    set('blink', eyes.blink * 0.8)

    for (const name of this.lastApplied.keys()) {
      if (!values.has(name)) this.expressionManager.setValue(name, 0)
    }
    for (const [name, val] of values) this.expressionManager.setValue(name, val)
    this.lastApplied = new Map(values)
  }

  _applyArkit(blendshapes) {
    for (const mesh of this.morphTargetMeshes) {
      const dict = mesh.morphTargetDictionary
      for (const [name, val] of Object.entries(blendshapes)) {
        if (name in dict) mesh.morphTargetInfluences[dict[name]] = Math.max(0, Math.min(1, val))
      }
    }
  }

  getDuration() {
    return this.animation ? this.animation.frames.length / this.animation.fps : 0
  }
}

const viewer = document.getElementById('viewer')
const loadingEl = document.getElementById('loading')
const textInput = document.getElementById('text-input')
const generateBtn = document.getElementById('generate-btn')
const stopBtn = document.getElementById('stop-btn')
const downloadBtn = document.getElementById('download-btn')
const statusEl = document.getElementById('status')
const timingEl = document.getElementById('timing')

let lastGeneratedAudio = null

const scene = new THREE.Scene()
scene.background = new THREE.Color(0x1a1a25)

const camera = new THREE.PerspectiveCamera(45, viewer.clientWidth / viewer.clientHeight, 0.1, 100)
camera.position.set(0, 1.2, 2.5)

const renderer = new THREE.WebGLRenderer({ antialias: true })
renderer.setSize(viewer.clientWidth, viewer.clientHeight)
renderer.setPixelRatio(window.devicePixelRatio)
renderer.outputColorSpace = THREE.SRGBColorSpace
viewer.appendChild(renderer.domElement)

const controls = new OrbitControls(camera, renderer.domElement)
controls.target.set(0, 1.0, 0)
controls.enableDamping = true
controls.dampingFactor = 0.05
controls.minDistance = 0.5
controls.maxDistance = 5
controls.update()

const ambient = new THREE.AmbientLight(0xffffff, 0.6)
scene.add(ambient)

const dirLight = new THREE.DirectionalLight(0xffffff, 1.2)
dirLight.position.set(2, 3, 2)
scene.add(dirLight)

const fillLight = new THREE.DirectionalLight(0x4488ff, 0.4)
fillLight.position.set(-2, 2, -2)
scene.add(fillLight)

const gridHelper = new THREE.GridHelper(4, 20, 0x303040, 0x252535)
scene.add(gridHelper)

let vrm = null
let facialPlayer = null

const gltfLoader = new GLTFLoader()
gltfLoader.register((parser) => new VRMLoaderPlugin(parser))

async function loadVRM() {
  console.log('[vrm] Starting to load Cleetus.vrm...')
  try {
    console.log('[vrm] Loading file from /Cleetus.vrm...')
    const gltf = await gltfLoader.loadAsync('/Cleetus.vrm',
      (xhr) => {
        const percent = (xhr.loaded / xhr.total * 100).toFixed(0)
        console.log(`[vrm] Loading progress: ${percent}% (${xhr.loaded}/${xhr.total})`)
      },
      (err) => {
        console.error('[vrm] Loading error:', err)
      }
    )
    console.log('[vrm] File loaded, extracting VRM data...')
    vrm = gltf.userData.vrm
    
    VRMUtils.removeUnnecessaryVertices(vrm.scene)
    VRMUtils.combineSkeletons(vrm.scene)
    
    vrm.scene.rotation.y = Math.PI
    vrm.scene.scale.setScalar(1.323)
    vrm.scene.position.y = -0.28
    
    vrm.scene.traverse((obj) => {
      if (obj.isMesh) {
        obj.castShadow = true
        obj.receiveShadow = true
      }
    })
    
    scene.add(vrm.scene)
    
    facialPlayer = new FacialAnimationPlayer(vrm)
    
    loadingEl.style.display = 'none'
    setStatus('Ready', 'ready')
  } catch (err) {
    console.error('Failed to load VRM:', err)
    console.error('Error stack:', err.stack)
    loadingEl.innerHTML = `<div style="color: #ff6b6b;">Failed to load VRM: ${err.message}<br><pre style="font-size:10px;text-align:left;margin-top:8px;">${err.stack}</pre></div>`
  }
}

function setStatus(text, cls = '') {
  statusEl.textContent = text
  statusEl.className = 'status ' + cls
}

generateBtn.addEventListener('click', async () => {
  if (!vrm || !facialPlayer) return
  
  const text = textInput.value.trim()
  if (!text) {
    setStatus('Please enter some text', 'error')
    return
  }
  
  generateBtn.disabled = true
  stopBtn.disabled = false
  setStatus('Generating...', 'loading')
  timingEl.textContent = ''
  
  const startTime = performance.now()
  
  try {
    const resp = await fetch('/api/generate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text })
    })
    
    if (!resp.ok) {
      const err = await resp.json()
      throw new Error(err.error || 'Generation failed')
    }
    
    const data = await resp.json()
    const genTime = ((performance.now() - startTime) / 1000).toFixed(1)
    const rtfx = (data.duration / parseFloat(genTime)).toFixed(1)
    
    await facialPlayer.load(data.animation, data.audio)
    facialPlayer.play()

    lastGeneratedAudio = data.audio
    downloadBtn.disabled = false

    setStatus(`Playing (${data.duration.toFixed(1)}s)`, 'ready')
    timingEl.textContent = `Generated in ${genTime}s (${rtfx}x realtime)`
  } catch (err) {
    console.error('Generation error:', err)
    setStatus(`Error: ${err.message}`, 'error')
  } finally {
    generateBtn.disabled = false
  }
})

stopBtn.addEventListener('click', () => {
  if (facialPlayer) {
    facialPlayer.stop()
    setStatus('Ready', 'ready')
  }
  stopBtn.disabled = true
})

downloadBtn.addEventListener('click', () => {
  if (!lastGeneratedAudio) return
  const link = document.createElement('a')
  link.href = `data:audio/wav;base64,${lastGeneratedAudio}`
  link.download = `cleetus_${Date.now()}.wav`
  document.body.appendChild(link)
  link.click()
  document.body.removeChild(link)
})

generateBtn.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault()
    generateBtn.click()
  }
})

textInput.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault()
    generateBtn.click()
  }
})

let lastTime = performance.now()
function animate() {
  requestAnimationFrame(animate)

  const now = performance.now()
  const deltaTime = (now - lastTime) / 1000
  lastTime = now

  if (vrm) {
    vrm.update(deltaTime)
  }

  if (facialPlayer) {
    facialPlayer.update(deltaTime)
  }

  controls.update()
  renderer.render(scene, camera)
}

window.addEventListener('resize', () => {
  camera.aspect = viewer.clientWidth / viewer.clientHeight
  camera.updateProjectionMatrix()
  renderer.setSize(viewer.clientWidth, viewer.clientHeight)
})

loadVRM()
animate()
