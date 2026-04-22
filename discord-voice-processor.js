import fs from 'fs'
import * as speakGate from './speak-gate.js'

let voiceReferencePath = null
let voiceReferenceText = null
let characterSystemPrompt = null
let characterName = 'assistant'

export function setCharacterCard(card) {
  const d = card.spec === 'chara_card_v2' ? card.data : card
  const name = d.name || 'the character'
  characterName = name
  const essence = [d.description, d.personality].filter(Boolean).join(' ')
  characterSystemPrompt = [
    `You are ${name}. Stay in character. ${essence}`,
    ``,
    `This is a live voice chat. You play ${name}. Reply with ${name}'s next spoken turn — the actual words ${name} would say out loud.`,
    ``,
    `Format rules (all strict):`,
    `- Output only the words ${name} speaks. Nothing else. No labels, no names, no colons.`,
    `- No narration, no actions in parentheses, no asterisks, no brackets, no stage directions.`,
    `- Do not write the other person's line or their name. Just your own turn, then stop.`,
    `- Do not repeat what the other person said.`,
    ``,
    `Length:`,
    `- Two or three full sentences is the target. A real conversational beat — you greet, you answer, you give a hook for them to come back.`,
    `- Minimum: one complete sentence of at least eight words. Never a single-word reply like "Amigo." or "Mira." alone — that reads as broken.`,
    `- Maximum: about forty words.`,
    ``,
    `Handling bad input:`,
    `- If the message is obviously noise, laughter, coughing, music, or gibberish, respond with a short in-character question that invites them to repeat — for example "eh, didn't catch that, say again?" — still a full sentence, still in character, and stop.`,
    ``,
    `Stop after one reply. One turn only. Never continue the conversation past your line.`,
  ].join('\n')
  console.log(`[processor] ✓ card loaded: ${name} | prompt=${characterSystemPrompt.length}ch`)
  speakGate.setCharacterCardPrompt(characterSystemPrompt)
}

export function getCharacterSystemPrompt() { return characterSystemPrompt }
export function getCharacterName() { return characterName }

function loadRefText(refAudioPath) {
  if (!refAudioPath) return null
  const lower = refAudioPath.toLowerCase()
  const sidecar = lower.endsWith('.wav') ? refAudioPath.slice(0, -4) + '.txt' : refAudioPath + '.txt'
  if (!fs.existsSync(sidecar)) {
    console.warn(`[processor] ⚠ no ref-text sidecar ${sidecar} — voice clone DISABLED`)
    return ''
  }
  const text = fs.readFileSync(sidecar, 'utf8').trim()
  console.log(`[processor] ref-text loaded (${text.length}ch)`)
  return text
}

export function setVoiceEmbedding(refAudioPath) {
  voiceReferencePath = refAudioPath
  voiceReferenceText = loadRefText(refAudioPath)
  console.log(`[processor] voice ref: ${refAudioPath}`)
  speakGate.setRefVoice(voiceReferencePath, voiceReferenceText || null)
}

export function getVoiceReferencePath() { return voiceReferencePath }
export function getVoiceReferenceText() { return voiceReferenceText }
export function clearHistory() { speakGate.clearHistory() }

export default { setVoiceEmbedding, setCharacterCard, getCharacterSystemPrompt, getVoiceReferenceText, getVoiceReferencePath, clearHistory }
