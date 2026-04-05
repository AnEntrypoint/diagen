import * as THREE from "three";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { VRMLoaderPlugin, VRMUtils } from "@pixiv/three-vrm";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { AnimationReader, mapVisemes, mapEyes } from "./animation-core.mjs";

class IdleAnimator {
	constructor(vrm) {
		this.vrm = vrm;
		this.time = 0;
		this.blinkState = {
			isBlinking: false,
			blinkTimer: 0,
			nextBlink: Math.random() * 2 + 2,
		};
		this.breathingPhase = Math.random() * Math.PI * 2;
		this.microMovements = {
			browPhase: Math.random() * Math.PI * 2,
			mouthPhase: Math.random() * Math.PI * 2,
			lookPhase: Math.random() * Math.PI * 2,
		};

		this.lookState = {
			target: { x: 0, y: 0 },
			current: { x: 0, y: 0 },
			nextSaccade: Math.random() * 1.5 + 0.5,
			saccadeTimer: 0,
			bigLookTimer: 0,
			nextBigLook: Math.random() * 4 + 3,
		};

		this.basePoses = new Map();
		this.posePhase = Math.random() * Math.PI * 2;
		this.sw = {
			phase: Math.random() * Math.PI * 2,
			speed: 0.5 + Math.random() * 0.3,
			magnitude: 0.03 + Math.random() * 0.02,
		};

		this.facialTargets = {
			browInnerUp: 0,
			browOuterUpLeft: 0,
			browOuterUpRight: 0,
			cheekSquintLeft: 0,
			cheekSquintRight: 0,
			noseSneerLeft: 0,
			noseSneerRight: 0,
			eyeWideLeft: 0,
			eyeWideRight: 0,
		};
		this.facialCurrent = { ...this.facialTargets };
		this.browTimer = 0;
		this.nextBrowChange = Math.random() * 2 + 1;
	}

	update(deltaTime) {
		this.time += deltaTime;

		const expressions = new Map();
		const clamp = (v, lo = 0, hi = 1) => Math.max(lo, Math.min(hi, v));

		this.breathingPhase += deltaTime * 1.5;
		const breathValue = clamp((Math.sin(this.breathingPhase) + 1) * 0.5 * 0.15);
		expressions.set("neutral", 1 - breathValue);

		this.microMovements.mouthPhase += deltaTime * 0.2;

		const cheekBase = (Math.sin(this.microMovements.mouthPhase * 0.7 + 1) + 1) * 0.5;
		const noseBase = (Math.sin(this.microMovements.mouthPhase * 0.5 + 2) + 1) * 0.5;

		this.facialTargets.cheekSquintLeft = cheekBase * 0.4;
		this.facialTargets.cheekSquintRight = cheekBase * 0.4;
		this.facialTargets.noseSneerLeft = noseBase * 0.35;
		this.facialTargets.noseSneerRight = noseBase * 0.35;

		this.browTimer += deltaTime;
		if (this.browTimer >= this.nextBrowChange) {
			this.browTimer = 0;
			this.nextBrowChange = Math.random() * 3 + 2;
			if (Math.random() > 0.4) {
				this.facialTargets.browInnerUp = Math.random() * 0.5 + 0.2;
				this.facialTargets.browOuterUpLeft = Math.random() * 0.3 + 0.1;
				this.facialTargets.browOuterUpRight = Math.random() * 0.3 + 0.1;
			} else {
				this.facialTargets.browInnerUp = 0;
				this.facialTargets.browOuterUpLeft = 0;
				this.facialTargets.browOuterUpRight = 0;
			}
		}

		const smoothSpeed = 2.5 * deltaTime;
		this.facialCurrent.browInnerUp += (this.facialTargets.browInnerUp - this.facialCurrent.browInnerUp) * smoothSpeed;
		this.facialCurrent.browOuterUpLeft += (this.facialTargets.browOuterUpLeft - this.facialCurrent.browOuterUpLeft) * smoothSpeed;
		this.facialCurrent.browOuterUpRight += (this.facialTargets.browOuterUpRight - this.facialCurrent.browOuterUpRight) * smoothSpeed;
		this.facialCurrent.cheekSquintLeft += (this.facialTargets.cheekSquintLeft - this.facialCurrent.cheekSquintLeft) * smoothSpeed;
		this.facialCurrent.cheekSquintRight += (this.facialTargets.cheekSquintRight - this.facialCurrent.cheekSquintRight) * smoothSpeed;
		this.facialCurrent.noseSneerLeft += (this.facialTargets.noseSneerLeft - this.facialCurrent.noseSneerLeft) * smoothSpeed;
		this.facialCurrent.noseSneerRight += (this.facialTargets.noseSneerRight - this.facialCurrent.noseSneerRight) * smoothSpeed;
		this.facialCurrent.eyeWideLeft += (this.facialTargets.eyeWideLeft - this.facialCurrent.eyeWideLeft) * smoothSpeed;
		this.facialCurrent.eyeWideRight += (this.facialTargets.eyeWideRight - this.facialCurrent.eyeWideRight) * smoothSpeed;

		expressions.set("browInnerUp", clamp(this.facialCurrent.browInnerUp));
		expressions.set("browOuterUpLeft", clamp(this.facialCurrent.browOuterUpLeft));
		expressions.set("browOuterUpRight", clamp(this.facialCurrent.browOuterUpRight));
		expressions.set("cheekSquintLeft", clamp(this.facialCurrent.cheekSquintLeft));
		expressions.set("cheekSquintRight", clamp(this.facialCurrent.cheekSquintRight));
		expressions.set("noseSneerLeft", clamp(this.facialCurrent.noseSneerLeft));
		expressions.set("noseSneerRight", clamp(this.facialCurrent.noseSneerRight));

		const eyeWideChance = Math.sin(this.time * 0.5) * 0.5 + 0.5;
		if (eyeWideChance > 0.92 && Math.random() > 0.99 && this.facialTargets.eyeWideLeft === 0) {
			this.facialTargets.eyeWideLeft = Math.random() * 0.8 + 0.2;
			this.facialTargets.eyeWideRight = Math.random() * 0.8 + 0.2;
			setTimeout(() => {
				this.facialTargets.eyeWideLeft = 0;
				this.facialTargets.eyeWideRight = 0;
			}, 250);
		}
		expressions.set("eyeWideLeft", clamp(this.facialCurrent.eyeWideLeft));
		expressions.set("eyeWideRight", clamp(this.facialCurrent.eyeWideRight));

		this.blinkState.blinkTimer += deltaTime;
		if (
			!this.blinkState.isBlinking &&
			this.blinkState.blinkTimer >= this.blinkState.nextBlink
		) {
			this.blinkState.isBlinking = true;
			this.blinkState.blinkTimer = 0;
		}
		if (this.blinkState.isBlinking) {
			const blinkProgress = this.blinkState.blinkTimer / 0.15;
			if (blinkProgress >= 1) {
				this.blinkState.isBlinking = false;
				this.blinkState.blinkTimer = 0;
				this.blinkState.nextBlink = Math.random() * 3 + 2;
			} else {
				const blinkValue = Math.sin(blinkProgress * Math.PI);
				expressions.set("blink", clamp(blinkValue));
				expressions.set("blinkLeft", clamp(blinkValue));
				expressions.set("blinkRight", clamp(blinkValue));
			}
		}

		this.lookState.saccadeTimer += deltaTime;
		this.lookState.bigLookTimer += deltaTime;

		if (this.lookState.bigLookTimer >= this.lookState.nextBigLook) {
			this.lookState.target.x = (Math.random() - 0.5) * 1.2;
			this.lookState.target.y = (Math.random() - 0.5) * 0.8;
			this.lookState.bigLookTimer = 0;
			this.lookState.nextBigLook = Math.random() * 4 + 3;
		} else if (this.lookState.saccadeTimer >= this.lookState.nextSaccade) {
			const saccadeStrength = Math.random() > 0.7 ? 0.4 : 0.15;
			this.lookState.target.x = clamp(
				this.lookState.target.x + (Math.random() - 0.5) * saccadeStrength,
				-0.8,
				0.8,
			);
			this.lookState.target.y = clamp(
				this.lookState.target.y + (Math.random() - 0.5) * saccadeStrength * 0.6,
				-0.5,
				0.5,
			);
			this.lookState.saccadeTimer = 0;
			this.lookState.nextSaccade = Math.random() * 0.8 + 0.2;
		}

		const lookSpeed = 3.0;
		this.lookState.current.x +=
			(this.lookState.target.x - this.lookState.current.x) *
			lookSpeed *
			deltaTime;
		this.lookState.current.y +=
			(this.lookState.target.y - this.lookState.current.y) *
			lookSpeed *
			deltaTime;

		const lookX = this.lookState.current.x;
		const lookY = this.lookState.current.y;

		if (lookX > 0.1) {
			expressions.set("lookRight", clamp(lookX));
			expressions.set("lookLeft", 0);
		} else if (lookX < -0.1) {
			expressions.set("lookLeft", clamp(-lookX));
			expressions.set("lookRight", 0);
		} else {
			expressions.set("lookLeft", 0);
			expressions.set("lookRight", 0);
		}

		if (lookY > 0.1) {
			expressions.set("lookUp", clamp(lookY));
			expressions.set("lookDown", 0);
		} else if (lookY < -0.1) {
			expressions.set("lookDown", clamp(-lookY));
			expressions.set("lookUp", 0);
		} else {
			expressions.set("lookUp", 0);
			expressions.set("lookDown", 0);
		}

		this._updateBonePoses(deltaTime);

		return expressions;
	}

	saveBasePose(humanoid) {
		const bones = [
			"leftUpperArm",
			"rightUpperArm",
			"leftLowerArm",
			"rightLowerArm",
			"spine",
			"chest",
			"neck",
		];
		for (const name of bones) {
			const bone = humanoid.getNormalizedBoneNode(name);
			if (bone) {
				this.basePoses.set(name, {
					x: bone.rotation.x,
					y: bone.rotation.y,
					z: bone.rotation.z,
				});
			}
		}
	}

	_updateBonePoses(deltaTime) {
		if (!this.vrm.humanoid || this.basePoses.size === 0) return;

		const humanoid = this.vrm.humanoid;

		this.posePhase += deltaTime * 0.8;

		const getBase = (name) => this.basePoses.get(name) || { x: 0, y: 0, z: 0 };
		const setRot = (name, x, y, z) => {
			const bone = humanoid.getNormalizedBoneNode(name);
			if (bone) bone.rotation.set(x, y, z);
		};

		const base = getBase("spine");
		const sway = Math.sin(this.posePhase) * this.sw.magnitude;
		setRot(
			"spine",
			base.x + sway * 0.5,
			base.y + Math.cos(this.posePhase * 0.7) * 0.01,
			base.z,
		);

		const chestBase = getBase("chest");
		setRot("chest", chestBase.x + sway * 0.3, chestBase.y, chestBase.z);

		const neckBase = getBase("neck");
		const headSway = Math.sin(this.posePhase * 0.6 + 1) * 0.02;
		setRot(
			"neck",
			neckBase.x + headSway,
			neckBase.y + Math.cos(this.posePhase * 0.5) * 0.015,
			neckBase.z,
		);

		const lArmBase = getBase("leftUpperArm");
		const rArmBase = getBase("rightUpperArm");
		const armSway = Math.sin(this.posePhase * 0.4) * 0.02;
		setRot(
			"leftUpperArm",
			lArmBase.x + armSway * 0.5,
			lArmBase.y,
			lArmBase.z + armSway * 0.3,
		);
		setRot(
			"rightUpperArm",
			rArmBase.x + armSway * 0.5,
			rArmBase.y,
			rArmBase.z - armSway * 0.3,
		);
	}
}

class FacialAnimationPlayer {
	constructor(vrm) {
		this.vrm = vrm;
		this.animation = null;
		this.audio = null;
		this.isPlaying = false;
		this.startTime = 0;
		this.availableExpressions = new Set();
		this.lastApplied = new Map();
		this.morphTargetMeshes = [];
		this.idleAnimator = new IdleAnimator(vrm);
		this.currentExpressions = new Map();

		vrm.scene.traverse((obj) => {
			if (obj.isMesh && obj.morphTargetDictionary)
				this.morphTargetMeshes.push(obj);
		});

		if (vrm.blendShapeProxy) {
			this.vrmVersion = "0.0";
			this.blendProxy = vrm.blendShapeProxy;
			const presets = [
				"A",
				"I",
				"U",
				"E",
				"O",
				"Blink",
				"Blink_L",
				"Blink_R",
				"Neutral",
				"LookUp",
				"LookDown",
				"LookLeft",
				"LookRight",
			];
			presets.forEach((p) => this.availableExpressions.add(p));
		} else if (vrm.expressionManager) {
			this.vrmVersion = "1.0";
			this.expressionManager = vrm.expressionManager;
			vrm.expressionManager.expressions.forEach((e) =>
				this.availableExpressions.add(e.expressionName),
			);
		} else {
			this.vrmVersion = "arkit";
		}

		console.log(
			`[vrm] Detected VRM ${this.vrmVersion}, expressions:`,
			[...this.availableExpressions].join(", ") || "morph targets",
		);
	}

	loadAnimation(buffer) {
		this.animation = new AnimationReader().fromBuffer(buffer);
		return this.animation;
	}

	loadAudio(base64) {
		this.audio = new Audio();
		this.audio.src = `data:audio/wav;base64,${base64}`;
		return new Promise((resolve, reject) => {
			this.audio.oncanplaythrough = () => resolve(this);
			this.audio.onerror = () => reject(new Error("Failed to load audio"));
		});
	}

	async load(animBase64, audioBase64) {
		if (animBase64) {
			const animBuffer = Uint8Array.from(atob(animBase64), (c) =>
				c.charCodeAt(0),
			).buffer;
			this.loadAnimation(animBuffer);
		}
		if (audioBase64) {
			await this.loadAudio(audioBase64);
		}
		return this;
	}

	play() {
		if (!this.animation) return;
		this.isPlaying = true;
		this.startTime = performance.now();
		this.lastApplied.clear();

		if (this.audio) {
			this.audio.currentTime = 0;
			this.audio.play().catch(() => {});
		}
	}

	stop() {
		this.isPlaying = false;
		if (this.audio) {
			this.audio.pause();
			this.audio.currentTime = 0;
		}
		this.resetExpressions();
	}

	resetExpressions() {
		this.currentExpressions.clear();
		if (this.vrmVersion === "0.0") {
			for (const name of this.availableExpressions)
				this.blendProxy.setValue(name, 0);
		} else if (this.vrmVersion === "1.0") {
			for (const name of this.availableExpressions)
				this.expressionManager.setValue(name, 0);
		} else {
			for (const mesh of this.morphTargetMeshes) {
				if (mesh.morphTargetInfluences) mesh.morphTargetInfluences.fill(0);
			}
		}
		this.lastApplied.clear();
	}

	update(deltaTime = 1 / 60) {
		if (this.isPlaying && this.animation) {
			const elapsed = (performance.now() - this.startTime) / 1000;
			const frame = this.animation.getFrameAtTime(elapsed);

			const idleExpressions = this.idleAnimator.update(deltaTime);

			if (frame) {
				this.applyFrame(frame.blendshapes, idleExpressions);
			}
			if (elapsed >= this.animation.frames.length / this.animation.fps) {
				this.isPlaying = false;
				this.resetExpressions();
			}
		} else {
			const idleExpressions = this.idleAnimator.update(deltaTime);
			this._applyIdleExpressions(idleExpressions);
		}
	}

	_applyIdleExpressions(idleExpressions) {
		const values = new Map();
		const has = (n) => this.availableExpressions.has(n);

		if (this.vrmVersion === "1.0") {
			for (const [name, val] of idleExpressions) {
				if (has(name)) values.set(name, val);
			}
			for (const [name, val] of values) {
				this.expressionManager.setValue(name, val);
			}
			for (const name of this.lastApplied.keys()) {
				if (!values.has(name)) this.expressionManager.setValue(name, 0);
			}
			this.lastApplied = new Map(values);

			this.morphTargetMeshes.forEach((mesh) => {
				if (!mesh.morphTargetDictionary || !mesh.morphTargetInfluences) return;
				for (const [name, val] of idleExpressions) {
					if (name in mesh.morphTargetDictionary) {
						mesh.morphTargetInfluences[mesh.morphTargetDictionary[name]] = val;
					}
				}
			});
		} else if (this.vrmVersion === "0.0") {
			for (const [name, val] of idleExpressions) {
				const vrm0Name =
					name === "blink"
						? "Blink"
						: name === "blinkLeft"
							? "Blink_L"
							: name === "blinkRight"
								? "Blink_R"
								: name === "neutral"
									? "Neutral"
									: name === "lookUp"
										? "LookUp"
										: name === "lookDown"
											? "LookDown"
											: name === "lookLeft"
												? "LookLeft"
												: name === "lookRight"
													? "LookRight"
													: name;
				if (has(vrm0Name)) this.blendProxy.setValue(vrm0Name, val);
			}

			this.morphTargetMeshes.forEach((mesh) => {
				if (!mesh.morphTargetDictionary || !mesh.morphTargetInfluences) return;
				for (const [name, val] of idleExpressions) {
					if (name in mesh.morphTargetDictionary) {
						mesh.morphTargetInfluences[mesh.morphTargetDictionary[name]] = val;
					}
				}
			});
		}
	}

	applyFrame(blendshapes, idleExpressions) {
		if (this.vrmVersion === "0.0")
			return this._applyVrm0(blendshapes, idleExpressions);
		if (this.vrmVersion === "1.0")
			return this._applyVrm1(blendshapes, idleExpressions);
		this._applyArkit(blendshapes);
	}

	_dominantViseme(visemes) {
		return Object.entries(visemes).reduce(
			(a, b) => (b[1] > a[1] ? b : a),
			["", 0],
		);
	}

	_applyVrm0(blendshapes, idleExpressions) {
		const clamp = (v, lo = 0, hi = 1) => Math.max(lo, Math.min(hi, v));
		const has = (n) => this.availableExpressions.has(n);
		const set = (n, v) => {
			if (has(n)) this.blendProxy.setValue(n, clamp(v));
		};

		const vrm0Visemes = { A: 0, I: 0, U: 0, E: 0, O: 0 };
		const v = mapVisemes(blendshapes);
		vrm0Visemes.A = v.aa;
		vrm0Visemes.I = v.ih;
		vrm0Visemes.U = v.ou;
		vrm0Visemes.E = v.ee;
		vrm0Visemes.O = v.oh;
		const [dom, domVal] = this._dominantViseme(vrm0Visemes);
		for (const k of ["A", "I", "U", "E", "O"]) set(k, k === dom ? domVal : 0);

		for (const [name, val] of idleExpressions) {
			const vrm0Name =
				name === "blink"
					? "Blink"
					: name === "blinkLeft"
						? "Blink_L"
						: name === "blinkRight"
							? "Blink_R"
							: name === "neutral"
								? "Neutral"
								: name === "lookUp"
									? "LookUp"
									: name === "lookDown"
										? "LookDown"
										: name === "lookLeft"
											? "LookLeft"
											: name === "lookRight"
												? "LookRight"
												: name;
			if (has(vrm0Name)) set(vrm0Name, val);
		}
	}

	_applyVrm1(blendshapes, idleExpressions) {
		const clamp = (v, lo = 0, hi = 1) => Math.max(lo, Math.min(hi, v));
		const values = new Map();
		const has = (n) => this.availableExpressions.has(n);
		const set = (n, v) => {
			if (has(n)) values.set(n, clamp(v));
		};

		const v = mapVisemes(blendshapes);
		const [dom, domVal] = this._dominantViseme(v);
		for (const k of ["aa", "ih", "ou", "ee", "oh"])
			set(k, k === dom ? domVal : 0);

		for (const [name, val] of idleExpressions) {
			if (has(name)) values.set(name, val);
		}

		for (const name of this.lastApplied.keys()) {
			if (!values.has(name)) this.expressionManager.setValue(name, 0);
		}
		for (const [name, val] of values)
			this.expressionManager.setValue(name, val);
		this.lastApplied = new Map(values);
	}

	_applyArkit(blendshapes) {
		for (const mesh of this.morphTargetMeshes) {
			const dict = mesh.morphTargetDictionary;
			for (const [name, val] of Object.entries(blendshapes)) {
				if (name in dict)
					mesh.morphTargetInfluences[dict[name]] = Math.max(
						0,
						Math.min(1, val),
					);
			}
		}
	}

	getDuration() {
		return this.animation
			? this.animation.frames.length / this.animation.fps
			: 0;
	}
}

const viewer = document.getElementById("viewer");
const loadingEl = document.getElementById("loading");
const textInput = document.getElementById("text-input");
const generateBtn = document.getElementById("generate-btn");
const stopBtn = document.getElementById("stop-btn");
const downloadBtn = document.getElementById("download-btn");
const replayBtn = document.getElementById("replay-btn");
const statusEl = document.getElementById("status");
const timingEl = document.getElementById("timing");

let lastGeneratedAudio = null;
let lastGeneratedAnimation = null;

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x1a1a25);

const camera = new THREE.PerspectiveCamera(
	45,
	viewer.clientWidth / viewer.clientHeight,
	0.1,
	100,
);
camera.position.set(0, 1.2, 2.5);

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(viewer.clientWidth, viewer.clientHeight);
renderer.setPixelRatio(window.devicePixelRatio);
renderer.outputColorSpace = THREE.SRGBColorSpace;
viewer.appendChild(renderer.domElement);

const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(0, 1.0, 0);
controls.enableDamping = true;
controls.dampingFactor = 0.05;
controls.minDistance = 0.5;
controls.maxDistance = 5;
controls.update();

const ambient = new THREE.AmbientLight(0xffffff, 0.6);
scene.add(ambient);

const dirLight = new THREE.DirectionalLight(0xffffff, 1.2);
dirLight.position.set(2, 3, 2);
scene.add(dirLight);

const fillLight = new THREE.DirectionalLight(0x4488ff, 0.4);
fillLight.position.set(-2, 2, -2);
scene.add(fillLight);

const gridHelper = new THREE.GridHelper(4, 20, 0x303040, 0x252535);
scene.add(gridHelper);

let vrm = null;
let facialPlayer = null;

const gltfLoader = new GLTFLoader();
gltfLoader.register((parser) => new VRMLoaderPlugin(parser));

async function loadVRM() {
	console.log("[vrm] Starting to load Cleetus.vrm...");
	try {
		console.log("[vrm] Loading file from /Cleetus.vrm...");
		const gltf = await gltfLoader.loadAsync(
			"/Cleetus.vrm",
			(xhr) => {
				const percent = ((xhr.loaded / xhr.total) * 100).toFixed(0);
				console.log(
					`[vrm] Loading progress: ${percent}% (${xhr.loaded}/${xhr.total})`,
				);
			},
			(err) => {
				console.error("[vrm] Loading error:", err);
			},
		);
		console.log("[vrm] File loaded, extracting VRM data...");
		vrm = gltf.userData.vrm;

		VRMUtils.removeUnnecessaryVertices(vrm.scene);
		VRMUtils.combineSkeletons(vrm.scene);

		vrm.scene.rotation.y = Math.PI;
		vrm.scene.scale.setScalar(1.323);
		vrm.scene.position.y = -0.28;

		vrm.scene.traverse((obj) => {
			if (obj.isMesh) {
				obj.castShadow = true;
				obj.receiveShadow = true;
			}
		});

		if (vrm.humanoid) {
			const humanoid = vrm.humanoid;
			const setBoneRot = (name, x, y, z) => {
				const bone = humanoid.getNormalizedBoneNode(name);
				if (bone) bone.rotation.set(x, y, z);
			};

			setBoneRot("leftUpperArm", -0, 0, 1.3);
			setBoneRot("rightUpperArm", -0, 0, -1.3);
			setBoneRot("leftLowerArm", -0.3, 0, 0);
			setBoneRot("rightLowerArm", -0.3, 0, 0);
			setBoneRot("leftHand", 0, -0.1, -0.1);
			setBoneRot("rightHand", 0, 0.1, 0.1);
			setBoneRot("spine", 0.02, 0, 0);
			setBoneRot("chest", 0.02, 0, 0);
			setBoneRot("neck", 0.05, 0, 0);
			setBoneRot("leftUpperLeg", 0.1, 0, -0.05);
			setBoneRot("rightUpperLeg", 0.1, 0, 0.05);
			setBoneRot("leftLowerLeg", -0.15, 0, 0);
			setBoneRot("rightLowerLeg", -0.15, 0, 0);
		}

		scene.add(vrm.scene);

		facialPlayer = new FacialAnimationPlayer(vrm);
		if (vrm.humanoid) {
			facialPlayer.idleAnimator.saveBasePose(vrm.humanoid);
		}

		loadingEl.style.display = "none";
		setStatus("Ready", "ready");
	} catch (err) {
		console.error("Failed to load VRM:", err);
		console.error("Error stack:", err.stack);
		loadingEl.innerHTML = `<div style="color: #ff6b6b;">Failed to load VRM: ${err.message}<br><pre style="font-size:10px;text-align:left;margin-top:8px;">${err.stack}</pre></div>`;
	}
}

function setStatus(text, cls = "") {
	statusEl.textContent = text;
	statusEl.className = "status " + cls;
}

generateBtn.addEventListener("click", async () => {
	if (!vrm || !facialPlayer) return;

	const text = textInput.value.trim();
	if (!text) {
		setStatus("Please enter some text", "error");
		return;
	}

	generateBtn.disabled = true;
	stopBtn.disabled = false;
	replayBtn.disabled = true;
	setStatus("Generating...", "loading");
	timingEl.textContent = "";

	const startTime = performance.now();

	try {
		const resp = await fetch("/api/generate", {
			method: "POST",
			headers: { "Content-Type": "application/json" },
			body: JSON.stringify({ text }),
		});

		if (!resp.ok) {
			const err = await resp.json();
			throw new Error(err.error || "Generation failed");
		}

		const data = await resp.json();
		const genTime = ((performance.now() - startTime) / 1000).toFixed(1);
		const rtfx = (data.duration / parseFloat(genTime)).toFixed(1);

		await facialPlayer.load(data.animation, data.audio);
		facialPlayer.play();

		lastGeneratedAudio = data.audio;
		lastGeneratedAnimation = data.animation;
		downloadBtn.disabled = false;
		replayBtn.disabled = false;

		setStatus(`Playing (${data.duration.toFixed(1)}s)`, "ready");
		timingEl.textContent = `Generated in ${genTime}s (${rtfx}x realtime)`;
	} catch (err) {
		console.error("Generation error:", err);
		setStatus(`Error: ${err.message}`, "error");
	} finally {
		generateBtn.disabled = false;
	}
});

stopBtn.addEventListener("click", () => {
	if (facialPlayer) {
		facialPlayer.stop();
		setStatus("Ready", "ready");
	}
	stopBtn.disabled = true;
});

downloadBtn.addEventListener("click", () => {
	if (!lastGeneratedAudio) return;
	const link = document.createElement("a");
	link.href = `data:audio/wav;base64,${lastGeneratedAudio}`;
	link.download = `cleetus_${Date.now()}.wav`;
	document.body.appendChild(link);
	link.click();
	document.body.removeChild(link);
});

replayBtn.addEventListener("click", async () => {
	if (!facialPlayer || !lastGeneratedAudio || !lastGeneratedAnimation) return;
	await facialPlayer.load(lastGeneratedAnimation, lastGeneratedAudio);
	facialPlayer.play();
	stopBtn.disabled = false;
});

generateBtn.addEventListener("keydown", (e) => {
	if (e.key === "Enter" && !e.shiftKey) {
		e.preventDefault();
		generateBtn.click();
	}
});

textInput.addEventListener("keydown", (e) => {
	if (e.key === "Enter" && !e.shiftKey) {
		e.preventDefault();
		generateBtn.click();
	}
});

let lastTime = performance.now();
function animate() {
	requestAnimationFrame(animate);

	const now = performance.now();
	const deltaTime = (now - lastTime) / 1000;
	lastTime = now;

	if (vrm) {
		vrm.update(deltaTime);
	}

	if (facialPlayer) {
		facialPlayer.update(deltaTime);
	}

	controls.update();
	renderer.render(scene, camera);
}

window.addEventListener("resize", () => {
	camera.aspect = viewer.clientWidth / viewer.clientHeight;
	camera.updateProjectionMatrix();
	renderer.setSize(viewer.clientWidth, viewer.clientHeight);
});

loadVRM();
animate();
