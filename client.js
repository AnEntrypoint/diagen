import * as THREE from 'three'
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js'
import { VRMLoaderPlugin, VRMUtils } from '@pixiv/three-vrm'
import { OrbitControls } from 'three/addons/controls/OrbitControls.js'
import { AnimationReader, mapVisemes, mapEyes, mapEmotions } from './animation-core.mjs'

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

    if (vrm.blendShapeProxy) {
      this.vrmVersion = '0.0'
      this.blendProxy = vrm.blendShapeProxy
      const presets = ['A','I','U','E','O','Blink','Blink_L','Blink_R','Joy','Angry','Sorrow','Fun','Neutral','LookUp','LookDown','LookLeft','LookRight']
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

  update() {
    if (!this.isPlaying || !this.animation) return
    
    const elapsed = (performance.now() - this.startTime) / 1000
    const frame = this.animation.getFrameAtTime(elapsed)
    if (!frame) return
    
    this.applyFrame(frame.blendshapes)
    
    if (elapsed >= this.animation.frames.length / this.animation.fps) {
      this.isPlaying = false
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
    set('Blink_L', eyes.blinkLeft)
    set('Blink_R', eyes.blinkRight)
    set('Blink', eyes.blink)
    set('LookUp', eyes.lookUp)
    set('LookDown', eyes.lookDown)
    set('LookLeft', eyes.lookLeft)
    set('LookRight', eyes.lookRight)

    const em = mapEmotions(blendshapes)
    set('Joy', em.happy)
    set('Sorrow', em.sad)
    set('Angry', em.angry)
    set('Fun', em.fun)
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
    set('blinkLeft', eyes.blinkLeft)
    set('blinkRight', eyes.blinkRight)
    set('blink', eyes.blink)
    set('lookUp', eyes.lookUp)
    set('lookDown', eyes.lookDown)
    set('lookLeft', eyes.lookLeft)
    set('lookRight', eyes.lookRight)

    const em = mapEmotions(blendshapes)
    set('happy', em.happy)
    set('sad', em.sad)
    set('angry', em.angry)
    set('relaxed', em.relaxed)
    set('surprised', em.surprised)

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
const statusEl = document.getElementById('status')
const timingEl = document.getElementById('timing')

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
  try {
    const gltf = await gltfLoader.loadAsync('/Cleetus.vrm')
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
    loadingEl.innerHTML = `<div style="color: #ff6b6b;">Failed to load VRM: ${err.message}</div>`
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

function animate() {
  requestAnimationFrame(animate)
  
  if (vrm) {
    vrm.update(1/60)
  }
  
  if (facialPlayer) {
    facialPlayer.update()
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
