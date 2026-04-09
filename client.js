import * as THREE from '/node_modules/three/build/three.module.js'
import { GLTFLoader } from '/node_modules/three/examples/jsm/loaders/GLTFLoader.js'
import { VRMLoaderPlugin, VRMUtils } from '/node_modules/@pixiv/three-vrm/lib/three-vrm.module.js'
import { OrbitControls } from '/node_modules/three/examples/jsm/controls/OrbitControls.js'
import { FacialAnimationPlayer } from '/facial-player.mjs'

const viewer = document.getElementById('viewer')
const loadingEl = document.getElementById('loading')
const textInput = document.getElementById('text-input')
const generateBtn = document.getElementById('generate-btn')
const stopBtn = document.getElementById('stop-btn')
const downloadBtn = document.getElementById('download-btn')
const replayBtn = document.getElementById('replay-btn')
const statusEl = document.getElementById('status')
const timingEl = document.getElementById('timing')
const promptInput = document.getElementById('prompt-input')
const askBtn = document.getElementById('ask-btn')
const llmStatusEl = document.getElementById('llm-status')

let lastGeneratedAudio = null, lastGeneratedAnimation = null
let vrm = null, facialPlayer = null, llmWorker = null

window.__debug = {
	get llmWorker() { return llmWorker },
	get vrm() { return vrm },
	get facialPlayer() { return facialPlayer },
}

function getLlmWorker() {
	if (llmWorker) return llmWorker
	llmWorker = new Worker('/llm-worker.js', { type: 'module' })
	llmStatusEl.textContent = 'Loading model...'
	askBtn.disabled = true
	llmWorker.postMessage({ type: 'load' })
	llmWorker.onmessage = (e) => {
		const { type, text, progress } = e.data
		if (type === 'progress') {
			llmStatusEl.textContent = text || `Loading ${Math.round((progress || 0) * 100)}%`
		} else if (type === 'loaded') {
			llmStatusEl.textContent = 'Model ready'
			askBtn.disabled = false
		} else if (type === 'chunk') {
			textInput.value += text
		} else if (type === 'done') {
			llmStatusEl.textContent = 'Done'
			askBtn.disabled = false
		} else if (type === 'error') {
			llmStatusEl.textContent = `Error: ${e.data.error}`
			askBtn.disabled = false
			throw new Error(`LLM worker error: ${e.data.error}`)
		}
	}
	return llmWorker
}

askBtn.addEventListener('click', () => {
	const prompt = promptInput.value.trim()
	if (!prompt) return
	askBtn.disabled = true
	textInput.value = ''
	llmStatusEl.textContent = 'Generating...'
	const w = getLlmWorker()
	w.postMessage({ type: 'generate', prompt })
})

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
scene.add(new THREE.AmbientLight(0xffffff, 0.6))
const dirLight = new THREE.DirectionalLight(0xffffff, 1.2)
dirLight.position.set(2, 3, 2)
scene.add(dirLight)
const fillLight = new THREE.DirectionalLight(0x4488ff, 0.4)
fillLight.position.set(-2, 2, -2)
scene.add(fillLight)
scene.add(new THREE.GridHelper(4, 20, 0x303040, 0x252535))

const gltfLoader = new GLTFLoader()
gltfLoader.register((parser) => new VRMLoaderPlugin(parser))

function setStatus(text, cls = '') { statusEl.textContent = text; statusEl.className = 'status ' + cls }

async function loadVRM() {
	const gltf = await gltfLoader.loadAsync('/Cleetus.vrm', (xhr) => {
		console.log(`[vrm] ${((xhr.loaded / xhr.total) * 100).toFixed(0)}%`)
	})
	vrm = gltf.userData.vrm
	VRMUtils.removeUnnecessaryVertices(vrm.scene)
	VRMUtils.combineSkeletons(vrm.scene)
	vrm.scene.rotation.y = Math.PI
	vrm.scene.scale.setScalar(1.323)
	vrm.scene.position.y = -0.28
	vrm.scene.traverse((obj) => { if (obj.isMesh) { obj.castShadow = true; obj.receiveShadow = true } })
	if (vrm.humanoid) {
		const h = vrm.humanoid
		const s = (n, x, y, z) => { const b = h.getNormalizedBoneNode(n); if (b) b.rotation.set(x, y, z) }
		s('leftUpperArm', 0, 0, 1.3); s('rightUpperArm', 0, 0, -1.3)
		s('leftLowerArm', -0.3, 0, 0); s('rightLowerArm', -0.3, 0, 0)
		s('leftHand', 0, -0.1, -0.1); s('rightHand', 0, 0.1, 0.1)
		s('spine', 0.02, 0, 0); s('chest', 0.02, 0, 0); s('neck', 0.05, 0, 0)
		s('leftUpperLeg', 0.1, 0, -0.05); s('rightUpperLeg', 0.1, 0, 0.05)
		s('leftLowerLeg', -0.15, 0, 0); s('rightLowerLeg', -0.15, 0, 0)
	}
	scene.add(vrm.scene)
	facialPlayer = new FacialAnimationPlayer(vrm)
	if (vrm.humanoid) facialPlayer.idleAnimator.saveBasePose(vrm.humanoid)
	loadingEl.style.display = 'none'
	setStatus('Ready', 'ready')
	askBtn.disabled = false
}

generateBtn.addEventListener('click', async () => {
	if (!vrm || !facialPlayer) return
	const text = textInput.value.trim()
	if (!text) { setStatus('Please enter some text', 'error'); return }
	generateBtn.disabled = true; stopBtn.disabled = false; replayBtn.disabled = true
	setStatus('Generating...', 'loading'); timingEl.textContent = ''
	const t0 = performance.now()
	try {
		const resp = await fetch('/api/generate', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ text }) })
		if (!resp.ok) { const err = await resp.json(); throw new Error(err.error || 'Generation failed') }
		const data = await resp.json()
		const genTime = ((performance.now() - t0) / 1000).toFixed(1)
		await facialPlayer.load(data.animation, data.audio)
		facialPlayer.play()
		lastGeneratedAudio = data.audio; lastGeneratedAnimation = data.animation
		downloadBtn.disabled = false; replayBtn.disabled = false
		setStatus(`Playing (${data.duration.toFixed(1)}s)`, 'ready')
		timingEl.textContent = `Generated in ${genTime}s (${(data.duration / parseFloat(genTime)).toFixed(1)}x realtime)`
	} catch (err) {
		console.error('Generation error:', err); setStatus(`Error: ${err.message}`, 'error')
	} finally { generateBtn.disabled = false }
})

stopBtn.addEventListener('click', () => { if (facialPlayer) { facialPlayer.stop(); setStatus('Ready', 'ready') }; stopBtn.disabled = true })

downloadBtn.addEventListener('click', () => {
	if (!lastGeneratedAudio) return
	const a = document.createElement('a')
	a.href = `data:audio/wav;base64,${lastGeneratedAudio}`
	a.download = `cleetus_${Date.now()}.wav`
	document.body.appendChild(a); a.click(); document.body.removeChild(a)
})

replayBtn.addEventListener('click', async () => {
	if (!facialPlayer || !lastGeneratedAudio || !lastGeneratedAnimation) return
	await facialPlayer.load(lastGeneratedAnimation, lastGeneratedAudio)
	facialPlayer.play(); stopBtn.disabled = false
})

textInput.addEventListener('keydown', (e) => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); generateBtn.click() } })

let lastTime = performance.now()
function animate() {
	requestAnimationFrame(animate)
	const now = performance.now(), dt = (now - lastTime) / 1000; lastTime = now
	if (vrm) vrm.update(dt)
	if (facialPlayer) facialPlayer.update(dt)
	controls.update(); renderer.render(scene, camera)
}

window.addEventListener('resize', () => {
	camera.aspect = viewer.clientWidth / viewer.clientHeight
	camera.updateProjectionMatrix()
	renderer.setSize(viewer.clientWidth, viewer.clientHeight)
})

loadVRM().catch((err) => {
	console.error('Failed to load VRM:', err)
	loadingEl.innerHTML = `<div style="color:#ff6b6b;">Failed to load VRM: ${err.message}</div>`
})
animate()
