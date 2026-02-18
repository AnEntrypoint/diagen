import * as THREE from 'three'
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js'
import { VRMLoaderPlugin, VRMUtils } from '@pixiv/three-vrm'
import { OrbitControls } from 'three/addons/controls/OrbitControls.js'

const MAGIC = 0x4146414E
const ARKIT_NAMES = [
  'browInnerUp', 'browDownLeft', 'browDownRight', 'browOuterUpLeft', 'browOuterUpRight',
  'eyeLookUpLeft', 'eyeLookUpRight', 'eyeLookDownLeft', 'eyeLookDownRight',
  'eyeLookInLeft', 'eyeLookInRight', 'eyeLookOutLeft', 'eyeLookOutRight',
  'eyeBlinkLeft', 'eyeBlinkRight', 'eyeSquintLeft', 'eyeSquintRight',
  'eyeWideLeft', 'eyeWideRight', 'cheekPuff', 'cheekSquintLeft', 'cheekSquintRight',
  'noseSneerLeft', 'noseSneerRight', 'jawOpen', 'jawForward', 'jawLeft', 'jawRight',
  'mouthFunnel', 'mouthPucker', 'mouthLeft', 'mouthRight',
  'mouthRollUpper', 'mouthRollLower', 'mouthShrugUpper', 'mouthShrugLower',
  'mouthOpen', 'mouthClose', 'mouthSmileLeft', 'mouthSmileRight',
  'mouthFrownLeft', 'mouthFrownRight', 'mouthDimpleLeft', 'mouthDimpleRight',
  'mouthUpperUpLeft', 'mouthUpperUpRight', 'mouthLowerDownLeft', 'mouthLowerDownRight',
  'mouthPressLeft', 'mouthPressRight', 'mouthStretchLeft', 'mouthStretchRight'
]

class AnimationReader {
  constructor() {
    this.fps = 30
    this.numBlendshapes = 0
    this.numFrames = 0
    this.names = ARKIT_NAMES
    this.frames = []
  }

  fromBuffer(buf) {
    let offset = 0
    const view = new DataView(buf instanceof ArrayBuffer ? buf : buf.buffer)
    
    const magic = view.getUint32(offset, true); offset += 4
    if (magic !== MAGIC) throw new Error('Invalid animation file')
    
    const version = view.getUint8(offset); offset += 1
    if (version < 1 || version > 2) throw new Error(`Unsupported version: ${version}`)
    
    this.fps = view.getUint8(offset); offset += 1
    this.numBlendshapes = view.getUint8(offset); offset += 1
    offset += 1
    this.numFrames = view.getUint32(offset, true); offset += 4
    
    if (version === 1) {
      this.names = []
      for (let i = 0; i < this.numBlendshapes; i++) {
        const len = view.getUint8(offset++)
        this.names.push(new TextDecoder().decode(new Uint8Array(buf, offset, len)))
        offset += len
      }
    }
    
    this.frames = []
    for (let f = 0; f < this.numFrames; f++) {
      const frame = {}
      for (let i = 0; i < this.numBlendshapes; i++) {
        frame[this.names[i]] = view.getUint8(offset++) / 255
      }
      this.frames.push({ time: f / this.fps, blendshapes: frame })
    }
    
    return this
  }

  getFrameAtTime(time) {
    const index = Math.floor(time * this.fps)
    return this.frames[Math.max(0, Math.min(index, this.frames.length - 1))]
  }
}

function mapVisemes(blendshapes) {
  const {
    jawOpen = 0, mouthClose = 0, mouthFunnel = 0, mouthPucker = 0,
    mouthSmileLeft = 0, mouthSmileRight = 0, mouthFrownLeft = 0, mouthFrownRight = 0,
    mouthUpperUpLeft = 0, mouthUpperUpRight = 0, mouthLowerDownLeft = 0, mouthLowerDownRight = 0,
    mouthStretchLeft = 0, mouthStretchRight = 0, mouthRollUpper = 0, mouthRollLower = 0,
    mouthPressLeft = 0, mouthPressRight = 0, mouthShrugUpper = 0, mouthShrugLower = 0,
    mouthDimpleLeft = 0, mouthDimpleRight = 0, mouthLeft = 0, mouthRight = 0
  } = blendshapes

  const clamp = (v, lo = 0, hi = 1) => Math.max(lo, Math.min(hi, v))
  const smile = Math.max(mouthSmileLeft, mouthSmileRight)
  const stretch = Math.max(mouthStretchLeft, mouthStretchRight)
  const upperUp = Math.max(mouthUpperUpLeft, mouthUpperUpRight)
  const lowerDown = Math.max(mouthLowerDownLeft, mouthLowerDownRight)
  
  const aa = clamp(jawOpen * 0.7 + lowerDown * 0.3)
  const ih = clamp(upperUp * 0.6 + stretch * 0.4)
  const ou = clamp(mouthFunnel * 0.5 + mouthPucker * 0.5)
  const ee = clamp(stretch * 0.7 + (1 - jawOpen) * 0.3)
  const oh = clamp(mouthPucker * 0.4 + jawOpen * 0.4 + mouthFunnel * 0.2)

  return { aa, ih, ou, ee, oh }
}

function mapEyes(blendshapes) {
  const {
    eyeBlinkLeft = 0, eyeBlinkRight = 0, eyeSquintLeft = 0, eyeSquintRight = 0,
    eyeWideLeft = 0, eyeWideRight = 0, eyeLookUpLeft = 0, eyeLookUpRight = 0,
    eyeLookDownLeft = 0, eyeLookDownRight = 0, eyeLookInLeft = 0, eyeLookInRight = 0,
    eyeLookOutLeft = 0, eyeLookOutRight = 0
  } = blendshapes

  const clamp = (v, lo = 0, hi = 1) => Math.max(lo, Math.min(hi, v))
  return {
    blinkLeft: clamp(eyeBlinkLeft + eyeSquintLeft * 0.3),
    blinkRight: clamp(eyeBlinkRight + eyeSquintRight * 0.3),
    blink: clamp((eyeBlinkLeft + eyeBlinkRight) / 2),
    lookUp: clamp(Math.max(eyeLookUpLeft, eyeLookUpRight)),
    lookDown: clamp(Math.max(eyeLookDownLeft, eyeLookDownRight)),
    lookLeft: clamp(Math.max(eyeLookInLeft, eyeLookOutRight)),
    lookRight: clamp(Math.max(eyeLookInRight, eyeLookOutLeft))
  }
}

function mapEmotionsV1(blendshapes) {
  const {
    mouthSmileLeft = 0, mouthSmileRight = 0, mouthFrownLeft = 0, mouthFrownRight = 0,
    browInnerUp = 0, browDownLeft = 0, browDownRight = 0, browOuterUpLeft = 0, browOuterUpRight = 0,
    cheekPuff = 0, eyeSquintLeft = 0, eyeSquintRight = 0, noseSneerLeft = 0, noseSneerRight = 0,
    jawOpen = 0, mouthFunnel = 0, mouthPucker = 0, eyeWideLeft = 0, eyeWideRight = 0
  } = blendshapes

  const clamp = (v, lo = 0, hi = 1) => Math.max(lo, Math.min(hi, v))
  const smile = Math.max(mouthSmileLeft, mouthSmileRight)
  const frown = Math.max(mouthFrownLeft, mouthFrownRight)
  const browDown = Math.max(browDownLeft, browDownRight)
  const squint = Math.max(eyeSquintLeft, eyeSquintRight)
  const wide = Math.max(eyeWideLeft, eyeWideRight)
  const sneer = Math.max(noseSneerLeft, noseSneerRight)
  const browUp = browInnerUp + Math.max(browOuterUpLeft, browOuterUpRight)

  return {
    happy: clamp(smile * 0.9 + squint * 0.1),
    sad: clamp(frown * 0.6 + browDown * 0.3 + (1 - smile) * 0.1),
    angry: clamp(browDown * 0.5 + sneer * 0.3 + frown * 0.2),
    relaxed: clamp((1 - browDown) * 0.5 + smile * 0.3 + cheekPuff * 0.2),
    surprised: clamp(browUp * 0.6 + wide * 0.3 + jawOpen * 0.1)
  }
}

class FacialAnimationPlayer {
  constructor(vrm) {
    this.vrm = vrm
    this.expressionManager = vrm.expressionManager
    this.animation = null
    this.audio = null
    this.isPlaying = false
    this.startTime = 0
    this.availableExpressions = new Set()
    this.lastApplied = new Map()
    
    if (this.expressionManager) {
      this.expressionManager.expressions.forEach(e => {
        this.availableExpressions.add(e.expressionName)
      })
    }
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
    if (!this.expressionManager) return
    for (const name of this.availableExpressions) {
      this.expressionManager.setValue(name, 0)
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
    if (!this.expressionManager) return

    const clamp = (v, lo = 0, hi = 1) => Math.max(lo, Math.min(hi, v))
    const values = new Map()
    const has = (name) => this.availableExpressions.has(name)
    const set = (name, val) => {
      if (has(name) && val > 0.001) values.set(name, clamp(val))
    }

    const visemes = mapVisemes(blendshapes)
    set('aa', visemes.aa)
    set('ih', visemes.ih)
    set('ou', visemes.ou)
    set('ee', visemes.ee)
    set('oh', visemes.oh)

    const eyes = mapEyes(blendshapes)
    set('blinkLeft', eyes.blinkLeft)
    set('blinkRight', eyes.blinkRight)
    set('blink', eyes.blink)

    const emotions = mapEmotionsV1(blendshapes)
    set('happy', emotions.happy)
    set('sad', emotions.sad)
    set('angry', emotions.angry)
    set('relaxed', emotions.relaxed)
    set('surprised', emotions.surprised)

    for (const [name, val] of values) {
      this.expressionManager.setValue(name, val)
      this.lastApplied.set(name, val)
    }

    for (const name of this.lastApplied.keys()) {
      if (!values.has(name)) {
        const last = this.lastApplied.get(name)
        const decayed = last * 0.6
        if (decayed < 0.01) {
          this.expressionManager.setValue(name, 0)
          this.lastApplied.delete(name)
        } else {
          this.expressionManager.setValue(name, decayed)
          this.lastApplied.set(name, decayed)
        }
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
