import * as THREE from 'https://esm.sh/three@0.169.0'
import { GLTFLoader } from 'https://esm.sh/three@0.169.0/examples/jsm/loaders/GLTFLoader.js'
import { OrbitControls } from 'https://esm.sh/three@0.169.0/examples/jsm/controls/OrbitControls.js'
import { VRMLoaderPlugin, VRMUtils } from 'https://esm.sh/@pixiv/three-vrm@2.1.3'

class IdleAnimator {
  constructor(vrm) {
    this.vrm = vrm
    this.time = 0
    this.blinkState = { isBlinking: false, blinkTimer: 0, nextBlink: Math.random() * 2 + 2 }
    this.breathingPhase = Math.random() * Math.PI * 2
    this.microMovements = { browPhase: Math.random() * Math.PI * 2, mouthPhase: Math.random() * Math.PI * 2 }
    this.lookState = {
      target: { x: 0, y: 0 }, current: { x: 0, y: 0 },
      nextSaccade: Math.random() * 1.5 + 0.5, saccadeTimer: 0,
      bigLookTimer: 0, nextBigLook: Math.random() * 4 + 3
    }
    this.basePoses = new Map()
    this.posePhase = Math.random() * Math.PI * 2
    this.sw = { phase: Math.random() * Math.PI * 2, speed: 0.5 + Math.random() * 0.3, magnitude: 0.03 + Math.random() * 0.02 }
    this.facialTargets = { browInnerUp: 0, browOuterUpLeft: 0, browOuterUpRight: 0, cheekSquintLeft: 0, cheekSquintRight: 0, noseSneerLeft: 0, noseSneerRight: 0, eyeWideLeft: 0, eyeWideRight: 0 }
    this.facialCurrent = { ...this.facialTargets }
    this.browTimer = 0
    this.nextBrowChange = Math.random() * 2 + 1
  }

  saveBasePose(humanoid) {
    for (const name of ['leftUpperArm','rightUpperArm','leftLowerArm','rightLowerArm','spine','chest','neck']) {
      const bone = humanoid.getNormalizedBoneNode(name)
      if (bone) this.basePoses.set(name, { x: bone.rotation.x, y: bone.rotation.y, z: bone.rotation.z })
    }
  }

  update(deltaTime) {
    this.time += deltaTime
    const clamp = (v, lo = 0, hi = 1) => Math.max(lo, Math.min(hi, v))
    const expressions = new Map()

    this.breathingPhase += deltaTime * 1.5
    expressions.set('neutral', 1 - clamp((Math.sin(this.breathingPhase) + 1) * 0.5 * 0.15))

    this.microMovements.mouthPhase += deltaTime * 0.2
    this.facialTargets.cheekSquintLeft = this.facialTargets.cheekSquintRight = ((Math.sin(this.microMovements.mouthPhase * 0.7 + 1) + 1) * 0.5) * 0.4
    this.facialTargets.noseSneerLeft = this.facialTargets.noseSneerRight = ((Math.sin(this.microMovements.mouthPhase * 0.5 + 2) + 1) * 0.5) * 0.35

    this.browTimer += deltaTime
    if (this.browTimer >= this.nextBrowChange) {
      this.browTimer = 0; this.nextBrowChange = Math.random() * 3 + 2
      if (Math.random() > 0.4) {
        this.facialTargets.browInnerUp = Math.random() * 0.5 + 0.2
        this.facialTargets.browOuterUpLeft = this.facialTargets.browOuterUpRight = Math.random() * 0.3 + 0.1
      } else {
        this.facialTargets.browInnerUp = this.facialTargets.browOuterUpLeft = this.facialTargets.browOuterUpRight = 0
      }
    }

    const ss = 2.5 * deltaTime
    for (const k of Object.keys(this.facialCurrent)) {
      this.facialCurrent[k] += (this.facialTargets[k] - this.facialCurrent[k]) * ss
      expressions.set(k, clamp(this.facialCurrent[k]))
    }

    this.blinkState.blinkTimer += deltaTime
    if (!this.blinkState.isBlinking && this.blinkState.blinkTimer >= this.blinkState.nextBlink) {
      this.blinkState.isBlinking = true; this.blinkState.blinkTimer = 0
    }
    if (this.blinkState.isBlinking) {
      const p = this.blinkState.blinkTimer / 0.15
      if (p >= 1) { this.blinkState.isBlinking = false; this.blinkState.blinkTimer = 0; this.blinkState.nextBlink = Math.random() * 3 + 2 }
      else { const v = clamp(Math.sin(p * Math.PI)); expressions.set('blink', v); expressions.set('blinkLeft', v); expressions.set('blinkRight', v) }
    }

    this.lookState.saccadeTimer += deltaTime; this.lookState.bigLookTimer += deltaTime
    if (this.lookState.bigLookTimer >= this.lookState.nextBigLook) {
      this.lookState.target.x = (Math.random() - 0.5) * 1.4; this.lookState.target.y = (Math.random() - 0.5) * 1.0
      this.lookState.bigLookTimer = 0; this.lookState.nextBigLook = Math.random() * 2 + 1.5
    } else if (this.lookState.saccadeTimer >= this.lookState.nextSaccade) {
      const s = Math.random() > 0.7 ? 0.4 : 0.15
      this.lookState.target.x = clamp(this.lookState.target.x + (Math.random() - 0.5) * s, -0.8, 0.8)
      this.lookState.target.y = clamp(this.lookState.target.y + (Math.random() - 0.5) * s * 0.6, -0.5, 0.5)
      this.lookState.saccadeTimer = 0; this.lookState.nextSaccade = Math.random() * 0.8 + 0.2
    }

    const ls = 4.0
    this.lookState.current.x += (this.lookState.target.x - this.lookState.current.x) * ls * deltaTime
    this.lookState.current.y += (this.lookState.target.y - this.lookState.current.y) * ls * deltaTime
    const lx = this.lookState.current.x, ly = this.lookState.current.y, t = 0.05

    if (lx > t) { expressions.set('lookRight', clamp((lx-t)*1.5)); expressions.set('lookLeft', 0); expressions.set('eyeLookOutRight', clamp((lx-t)*0.8)); expressions.set('eyeLookInLeft', clamp((lx-t)*0.8)); expressions.set('eyeLookOutLeft', 0); expressions.set('eyeLookInRight', 0) }
    else if (lx < -t) { expressions.set('lookLeft', clamp((-lx-t)*1.5)); expressions.set('lookRight', 0); expressions.set('eyeLookOutLeft', clamp((-lx-t)*0.8)); expressions.set('eyeLookInRight', clamp((-lx-t)*0.8)); expressions.set('eyeLookOutRight', 0); expressions.set('eyeLookInLeft', 0) }
    else { for (const k of ['lookLeft','lookRight','eyeLookOutLeft','eyeLookOutRight','eyeLookInLeft','eyeLookInRight']) expressions.set(k, 0) }

    if (ly > t) { expressions.set('lookUp', clamp((ly-t)*1.5)); expressions.set('lookDown', 0); expressions.set('eyeLookUpLeft', clamp((ly-t)*0.9)); expressions.set('eyeLookUpRight', clamp((ly-t)*0.9)); expressions.set('eyeLookDownLeft', 0); expressions.set('eyeLookDownRight', 0) }
    else if (ly < -t) { expressions.set('lookDown', clamp((-ly-t)*1.5)); expressions.set('lookUp', 0); expressions.set('eyeLookDownLeft', clamp((-ly-t)*0.9)); expressions.set('eyeLookDownRight', clamp((-ly-t)*0.9)); expressions.set('eyeLookUpLeft', 0); expressions.set('eyeLookUpRight', 0) }
    else { for (const k of ['lookUp','lookDown','eyeLookUpLeft','eyeLookUpRight','eyeLookDownLeft','eyeLookDownRight']) expressions.set(k, 0) }

    this._updateBonePoses(deltaTime)
    return expressions
  }

  _updateBonePoses(deltaTime) {
    if (!this.vrm.humanoid || this.basePoses.size === 0) return
    const h = this.vrm.humanoid
    this.posePhase += deltaTime * 0.8
    const get = n => this.basePoses.get(n) || { x:0, y:0, z:0 }
    const set = (n, x, y, z) => { const b = h.getNormalizedBoneNode(n); if (b) b.rotation.set(x, y, z) }
    const sway = Math.sin(this.posePhase) * this.sw.magnitude
    const b = get('spine'); set('spine', b.x+sway*0.5, b.y+Math.cos(this.posePhase*0.7)*0.01, b.z)
    const c = get('chest'); set('chest', c.x+sway*0.3, c.y, c.z)
    const n = get('neck'); set('neck', n.x+Math.sin(this.posePhase*0.6+1)*0.02, n.y+Math.cos(this.posePhase*0.5)*0.015, n.z)
    const la = get('leftUpperArm'), ra = get('rightUpperArm'), as = Math.sin(this.posePhase*0.4)*0.02
    set('leftUpperArm', la.x+as*0.5, la.y, la.z+as*0.3); set('rightUpperArm', ra.x+as*0.5, ra.y, ra.z-as*0.3)
  }
}

let vrm = null, facialPlayer = null, mouthOpenTarget = 0, mouthOpenCurrent = 0

class VRMViewer {
  constructor(canvas) {
    this.canvas = canvas
    this.availableExpressions = new Set()
    this.expressionManager = null
    this.vrmVersion = null
    this.idleAnimator = null
    this.lastApplied = new Map()
  }

  async load(url) {
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0x0f0f13)

    const camera = new THREE.PerspectiveCamera(45, this.canvas.clientWidth / this.canvas.clientHeight, 0.1, 100)
    camera.position.set(0, 1.35, 2.0)

    const renderer = new THREE.WebGLRenderer({ canvas: this.canvas, antialias: true })
    renderer.setSize(this.canvas.clientWidth, this.canvas.clientHeight)
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
    renderer.outputColorSpace = THREE.SRGBColorSpace

    const controls = new OrbitControls(camera, this.canvas)
    controls.target.set(0, 1.15, 0); controls.enableDamping = true; controls.dampingFactor = 0.05
    controls.minDistance = 0.5; controls.maxDistance = 5; controls.update()

    scene.add(new THREE.AmbientLight(0xffffff, 0.6))
    const dir = new THREE.DirectionalLight(0xffffff, 1.2); dir.position.set(2, 3, 2); scene.add(dir)
    const fill = new THREE.DirectionalLight(0x4488ff, 0.4); fill.position.set(-2, 2, -2); scene.add(fill)
    scene.add(new THREE.GridHelper(4, 20, 0x303040, 0x252535))

    const loader = new GLTFLoader()
    loader.register(p => new VRMLoaderPlugin(p))
    const gltf = await loader.loadAsync(url)
    vrm = gltf.userData.vrm
    VRMUtils.removeUnnecessaryVertices(vrm.scene)
    VRMUtils.combineSkeletons(vrm.scene)
    vrm.scene.rotation.y = Math.PI
    vrm.scene.scale.setScalar(1.323)
    vrm.scene.position.y = -0.28
    scene.add(vrm.scene)

    if (vrm.expressionManager) {
      this.vrmVersion = '1.0'; this.expressionManager = vrm.expressionManager
      vrm.expressionManager.expressions.forEach(e => this.availableExpressions.add(e.expressionName))
    } else if (vrm.blendShapeProxy) {
      this.vrmVersion = '0.0'
      for (const p of ['A','I','U','E','O','Blink','Blink_L','Blink_R','Neutral','LookUp','LookDown','LookLeft','LookRight'])
        this.availableExpressions.add(p)
    }

    if (vrm.humanoid) {
      const h = vrm.humanoid
      const sb = (n, x, y, z) => { const b = h.getNormalizedBoneNode(n); if (b) b.rotation.set(x, y, z) }
      sb('leftUpperArm', 0, 0, 1.3); sb('rightUpperArm', 0, 0, -1.3)
      sb('leftLowerArm', -0.3, 0, 0); sb('rightLowerArm', -0.3, 0, 0)
      sb('spine', 0.02, 0, 0); sb('chest', 0.02, 0, 0); sb('neck', 0.05, 0, 0)
    }

    this.idleAnimator = new IdleAnimator(vrm)
    if (vrm.humanoid) this.idleAnimator.saveBasePose(vrm.humanoid)

    window.addEventListener('resize', () => {
      camera.aspect = this.canvas.clientWidth / this.canvas.clientHeight
      camera.updateProjectionMatrix()
      renderer.setSize(this.canvas.clientWidth, this.canvas.clientHeight)
    })

    let last = performance.now()
    const animate = () => {
      requestAnimationFrame(animate)
      const now = performance.now(), dt = Math.min((now - last) / 1000, 0.1); last = now
      vrm.update(dt)
      this._update(dt)
      controls.update()
      renderer.render(scene, camera)
    }
    animate()
    return this
  }

  _update(dt) {
    const clamp = (v, lo = 0, hi = 1) => Math.max(lo, Math.min(hi, v))
    const has = n => this.availableExpressions.has(n)
    const idle = this.idleAnimator.update(dt)

    mouthOpenCurrent += (mouthOpenTarget - mouthOpenCurrent) * Math.min(dt * 18, 1)
    const mouth = clamp(mouthOpenCurrent)

    if (this.vrmVersion === '1.0') {
      const vals = new Map(idle)
      if (has('aa')) vals.set('aa', mouth)
      for (const [n, v] of this.lastApplied.keys ? this.lastApplied : []) {
        if (!vals.has(n)) this.expressionManager.setValue(n, 0)
      }
      for (const [n, v] of vals) {
        if (has(n)) this.expressionManager.setValue(n, v)
      }
      this.lastApplied = vals
    } else if (this.vrmVersion === '0.0') {
      for (const [n, v] of idle) {
        const m = { blink:'Blink', blinkLeft:'Blink_L', blinkRight:'Blink_R', neutral:'Neutral', lookUp:'LookUp', lookDown:'LookDown', lookLeft:'LookLeft', lookRight:'LookRight' }
        const name = m[n] || n
        if (has(name)) vrm.blendShapeProxy.setValue(name, v)
      }
      if (has('A')) vrm.blendShapeProxy.setValue('A', mouth)
    }
  }

  setMouthOpen(v) {
    mouthOpenTarget = Math.max(0, Math.min(1, v))
  }
}

let viewer = null

export async function initVRM(canvas) {
  viewer = new VRMViewer(canvas)
  await viewer.load('./Cleetus.vrm')
  return viewer
}

export function setMouthOpen(v) {
  if (viewer) viewer.setMouthOpen(v)
}
