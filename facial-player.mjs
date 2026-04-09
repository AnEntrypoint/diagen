import { AnimationReader, mapVisemes } from './animation-core.mjs'
import { IdleAnimator } from './idle-animator.mjs'

const VRM0 = { blink: 'Blink', blinkLeft: 'Blink_L', blinkRight: 'Blink_R', neutral: 'Neutral', lookUp: 'LookUp', lookDown: 'LookDown', lookLeft: 'LookLeft', lookRight: 'LookRight' }

export class FacialAnimationPlayer {
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

		vrm.scene.traverse((obj) => { if (obj.isMesh && obj.morphTargetDictionary) this.morphTargetMeshes.push(obj) })

		if (vrm.blendShapeProxy) {
			this.vrmVersion = '0.0'
			this.blendProxy = vrm.blendShapeProxy
			for (const p of ['A', 'I', 'U', 'E', 'O', 'Blink', 'Blink_L', 'Blink_R', 'Neutral', 'LookUp', 'LookDown', 'LookLeft', 'LookRight'])
				this.availableExpressions.add(p)
		} else if (vrm.expressionManager) {
			this.vrmVersion = '1.0'
			this.expressionManager = vrm.expressionManager
			vrm.expressionManager.expressions.forEach((e) => this.availableExpressions.add(e.expressionName))
		} else {
			this.vrmVersion = 'arkit'
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
			const animBuffer = Uint8Array.from(atob(animBase64), (c) => c.charCodeAt(0)).buffer
			this.loadAnimation(animBuffer)
		}
		if (audioBase64) await this.loadAudio(audioBase64)
		return this
	}

	play() {
		if (!this.animation) return
		this.isPlaying = true
		this.startTime = performance.now()
		this.lastApplied.clear()
		if (this.audio) { this.audio.currentTime = 0; this.audio.play().catch(() => {}) }
	}

	stop() {
		this.isPlaying = false
		if (this.audio) { this.audio.pause(); this.audio.currentTime = 0 }
		this.resetExpressions()
	}

	resetExpressions() {
		this.currentExpressions.clear()
		if (this.vrmVersion === '0.0') {
			for (const name of this.availableExpressions) this.blendProxy.setValue(name, 0)
		} else if (this.vrmVersion === '1.0') {
			for (const name of this.availableExpressions) this.expressionManager.setValue(name, 0)
		} else {
			for (const mesh of this.morphTargetMeshes) { if (mesh.morphTargetInfluences) mesh.morphTargetInfluences.fill(0) }
		}
		this.lastApplied.clear()
	}

	update(deltaTime = 1 / 60) {
		const ie = this.idleAnimator.update(deltaTime)
		if (this.isPlaying && this.animation) {
			const elapsed = (performance.now() - this.startTime) / 1000
			const frame = this.animation.getFrameAtTime(elapsed)
			if (frame) this.applyFrame(frame.blendshapes, ie)
			if (elapsed >= this.animation.frames.length / this.animation.fps) {
				this.isPlaying = false
				this.resetExpressions()
			}
		} else {
			this._applyIdleExpressions(ie)
		}
	}

	_applyIdleExpressions(ie) {
		const has = (n) => this.availableExpressions.has(n)
		if (this.vrmVersion === '1.0') {
			const values = new Map()
			for (const [name, val] of ie) { if (has(name)) values.set(name, val) }
			for (const name of this.lastApplied.keys()) { if (!values.has(name)) this.expressionManager.setValue(name, 0) }
			for (const [name, val] of values) this.expressionManager.setValue(name, val)
			this.lastApplied = new Map(values)
			for (const mesh of this.morphTargetMeshes) {
				if (!mesh.morphTargetDictionary || !mesh.morphTargetInfluences) continue
				for (const [name, val] of ie) { if (name in mesh.morphTargetDictionary) mesh.morphTargetInfluences[mesh.morphTargetDictionary[name]] = val }
			}
		} else if (this.vrmVersion === '0.0') {
			for (const [name, val] of ie) {
				const n = VRM0[name] || name
				if (has(n)) this.blendProxy.setValue(n, val)
			}
			for (const mesh of this.morphTargetMeshes) {
				if (!mesh.morphTargetDictionary || !mesh.morphTargetInfluences) continue
				for (const [name, val] of ie) { if (name in mesh.morphTargetDictionary) mesh.morphTargetInfluences[mesh.morphTargetDictionary[name]] = val }
			}
		}
	}

	applyFrame(blendshapes, ie) {
		if (this.vrmVersion === '0.0') return this._applyVrm0(blendshapes, ie)
		if (this.vrmVersion === '1.0') return this._applyVrm1(blendshapes, ie)
		this._applyArkit(blendshapes)
	}

	_dominantViseme(v) {
		return Object.entries(v).reduce((a, b) => (b[1] > a[1] ? b : a), ['', 0])
	}

	_applyVrm0(blendshapes, ie) {
		const clamp = (v) => Math.max(0, Math.min(1, v))
		const has = (n) => this.availableExpressions.has(n)
		const set = (n, v) => { if (has(n)) this.blendProxy.setValue(n, clamp(v)) }
		const vmap = mapVisemes(blendshapes)
		const vrm0v = { A: vmap.aa, I: vmap.ih, U: vmap.ou, E: vmap.ee, O: vmap.oh }
		const [dom, domVal] = this._dominantViseme(vrm0v)
		for (const k of ['A', 'I', 'U', 'E', 'O']) set(k, k === dom ? domVal : 0)
		for (const [name, val] of ie) { const n = VRM0[name] || name; if (has(n)) set(n, val) }
	}

	_applyVrm1(blendshapes, ie) {
		const clamp = (v) => Math.max(0, Math.min(1, v))
		const values = new Map()
		const has = (n) => this.availableExpressions.has(n)
		const set = (n, v) => { if (has(n)) values.set(n, clamp(v)) }
		const v = mapVisemes(blendshapes)
		const [dom, domVal] = this._dominantViseme(v)
		for (const k of ['aa', 'ih', 'ou', 'ee', 'oh']) set(k, k === dom ? domVal : 0)
		for (const [name, val] of ie) { if (has(name)) values.set(name, val) }
		for (const name of this.lastApplied.keys()) { if (!values.has(name)) this.expressionManager.setValue(name, 0) }
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
