import { clamp } from './animation-core.mjs'

export class IdleAnimator {
	constructor(vrm) {
		this.vrm = vrm
		this.time = 0
		this.blinkState = { isBlinking: false, blinkTimer: 0, nextBlink: Math.random() * 2 + 2 }
		this.breathingPhase = Math.random() * Math.PI * 2
		this.microMovements = { browPhase: Math.random() * Math.PI * 2, mouthPhase: Math.random() * Math.PI * 2, lookPhase: Math.random() * Math.PI * 2 }
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

	update(deltaTime) {
		this.time += deltaTime
		const expressions = new Map()

		this.breathingPhase += deltaTime * 1.5
		const breathValue = clamp((Math.sin(this.breathingPhase) + 1) * 0.5 * 0.15)
		expressions.set('neutral', 1 - breathValue)

		this.microMovements.mouthPhase += deltaTime * 0.2
		const cheekBase = (Math.sin(this.microMovements.mouthPhase * 0.7 + 1) + 1) * 0.5
		const noseBase = (Math.sin(this.microMovements.mouthPhase * 0.5 + 2) + 1) * 0.5
		this.facialTargets.cheekSquintLeft = this.facialTargets.cheekSquintRight = cheekBase * 0.4
		this.facialTargets.noseSneerLeft = this.facialTargets.noseSneerRight = noseBase * 0.35

		this.browTimer += deltaTime
		if (this.browTimer >= this.nextBrowChange) {
			this.browTimer = 0
			this.nextBrowChange = Math.random() * 3 + 2
			const raise = Math.random() > 0.4
			this.facialTargets.browInnerUp = raise ? Math.random() * 0.5 + 0.2 : 0
			this.facialTargets.browOuterUpLeft = this.facialTargets.browOuterUpRight = raise ? Math.random() * 0.3 + 0.1 : 0
		}

		const ss = 2.5 * deltaTime
		for (const k of Object.keys(this.facialTargets)) {
			this.facialCurrent[k] += (this.facialTargets[k] - this.facialCurrent[k]) * ss
			expressions.set(k, clamp(this.facialCurrent[k]))
		}

		const eyeWideChance = Math.sin(this.time * 0.5) * 0.5 + 0.5
		if (eyeWideChance > 0.92 && Math.random() > 0.99 && this.facialTargets.eyeWideLeft === 0) {
			this.facialTargets.eyeWideLeft = this.facialTargets.eyeWideRight = Math.random() * 0.8 + 0.2
			setTimeout(() => { this.facialTargets.eyeWideLeft = this.facialTargets.eyeWideRight = 0 }, 250)
		}

		this.blinkState.blinkTimer += deltaTime
		if (!this.blinkState.isBlinking && this.blinkState.blinkTimer >= this.blinkState.nextBlink) {
			this.blinkState.isBlinking = true
			this.blinkState.blinkTimer = 0
		}
		if (this.blinkState.isBlinking) {
			const bp = this.blinkState.blinkTimer / 0.15
			if (bp >= 1) {
				this.blinkState.isBlinking = false
				this.blinkState.blinkTimer = 0
				this.blinkState.nextBlink = Math.random() * 3 + 2
			} else {
				const bv = Math.sin(bp * Math.PI)
				expressions.set('blink', clamp(bv))
				expressions.set('blinkLeft', clamp(bv))
				expressions.set('blinkRight', clamp(bv))
			}
		}

		this.lookState.saccadeTimer += deltaTime
		this.lookState.bigLookTimer += deltaTime
		if (this.lookState.bigLookTimer >= this.lookState.nextBigLook) {
			this.lookState.target.x = (Math.random() - 0.5) * 1.4
			this.lookState.target.y = (Math.random() - 0.5) * 1.0
			this.lookState.bigLookTimer = 0
			this.lookState.nextBigLook = Math.random() * 2 + 1.5
		} else if (this.lookState.saccadeTimer >= this.lookState.nextSaccade) {
			const sc = Math.random() > 0.7 ? 0.4 : 0.15
			this.lookState.target.x = clamp(this.lookState.target.x + (Math.random() - 0.5) * sc, -0.8, 0.8)
			this.lookState.target.y = clamp(this.lookState.target.y + (Math.random() - 0.5) * sc * 0.6, -0.5, 0.5)
			this.lookState.saccadeTimer = 0
			this.lookState.nextSaccade = Math.random() * 0.8 + 0.2
		}

		const ls = 4.0
		this.lookState.current.x += (this.lookState.target.x - this.lookState.current.x) * ls * deltaTime
		this.lookState.current.y += (this.lookState.target.y - this.lookState.current.y) * ls * deltaTime

		const lx = this.lookState.current.x, ly = this.lookState.current.y, thr = 0.05
		const lxPos = clamp((lx - thr) * 1.5), lxNeg = clamp((-lx - thr) * 1.5)
		const lxPosE = clamp((lx - thr) * 0.8), lxNegE = clamp((-lx - thr) * 0.8)
		expressions.set('lookRight', lx > thr ? lxPos : 0)
		expressions.set('lookLeft', lx < -thr ? lxNeg : 0)
		expressions.set('eyeLookOutRight', lx > thr ? lxPosE : 0)
		expressions.set('eyeLookInLeft', lx > thr ? lxPosE : 0)
		expressions.set('eyeLookOutLeft', lx < -thr ? lxNegE : 0)
		expressions.set('eyeLookInRight', lx < -thr ? lxNegE : 0)
		const lyPos = clamp((ly - thr) * 1.5), lyNeg = clamp((-ly - thr) * 1.5)
		const lyPosE = clamp((ly - thr) * 0.9), lyNegE = clamp((-ly - thr) * 0.9)
		expressions.set('lookUp', ly > thr ? lyPos : 0)
		expressions.set('lookDown', ly < -thr ? lyNeg : 0)
		expressions.set('eyeLookUpLeft', ly > thr ? lyPosE : 0)
		expressions.set('eyeLookUpRight', ly > thr ? lyPosE : 0)
		expressions.set('eyeLookDownLeft', ly < -thr ? lyNegE : 0)
		expressions.set('eyeLookDownRight', ly < -thr ? lyNegE : 0)

		this._updateBonePoses(deltaTime)
		return expressions
	}

	saveBasePose(humanoid) {
		for (const name of ['leftUpperArm', 'rightUpperArm', 'leftLowerArm', 'rightLowerArm', 'spine', 'chest', 'neck']) {
			const bone = humanoid.getNormalizedBoneNode(name)
			if (bone) this.basePoses.set(name, { x: bone.rotation.x, y: bone.rotation.y, z: bone.rotation.z })
		}
	}

	_updateBonePoses(deltaTime) {
		if (!this.vrm.humanoid || this.basePoses.size === 0) return
		const h = this.vrm.humanoid
		this.posePhase += deltaTime * 0.8
		const get = (n) => this.basePoses.get(n) || { x: 0, y: 0, z: 0 }
		const set = (n, x, y, z) => { const b = h.getNormalizedBoneNode(n); if (b) b.rotation.set(x, y, z) }
		const sway = Math.sin(this.posePhase) * this.sw.magnitude
		const s = get('spine'); set('spine', s.x + sway * 0.5, s.y + Math.cos(this.posePhase * 0.7) * 0.01, s.z)
		const c = get('chest'); set('chest', c.x + sway * 0.3, c.y, c.z)
		const nk = get('neck'); const hs = Math.sin(this.posePhase * 0.6 + 1) * 0.02
		set('neck', nk.x + hs, nk.y + Math.cos(this.posePhase * 0.5) * 0.015, nk.z)
		const la = get('leftUpperArm'), ra = get('rightUpperArm')
		const as = Math.sin(this.posePhase * 0.4) * 0.02
		set('leftUpperArm', la.x + as * 0.5, la.y, la.z + as * 0.3)
		set('rightUpperArm', ra.x + as * 0.5, ra.y, ra.z - as * 0.3)
	}
}
