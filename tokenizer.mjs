export function buildTokenizer(tokJson) {
  const { model, added_tokens = [] } = tokJson
  const vocab = model.vocab
  const idToToken = Object.fromEntries(Object.entries(vocab).map(([t, id]) => [id, t]))
  for (const at of added_tokens) idToToken[at.id] = at.content
  const addedMap = Object.fromEntries(added_tokens.map(at => [at.content, at.id]))
  const addedList = added_tokens.map(at => at.content).sort((a, b) => b.length - a.length)
  const merges = model.merges.map(m => Array.isArray(m) ? m : m.split(' '))
  const byteEnc = {}
  let n = 0
  for (let b = 0; b < 256; b++) {
    if ((b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174 && b <= 255)) byteEnc[b] = String.fromCharCode(b)
    else byteEnc[b] = String.fromCharCode(256 + n++)
  }
  const byteDec = Object.fromEntries(Object.entries(byteEnc).map(([b, c]) => [c, +b]))
  function bpe(word) {
    let w = [...word]
    while (w.length > 1) {
      let best = -1, bestPair = null
      for (let i = 0; i < w.length - 1; i++) {
        const idx = merges.findIndex(m => m[0] === w[i] && m[1] === w[i + 1])
        if (idx >= 0 && (best < 0 || idx < best)) { best = idx; bestPair = i }
      }
      if (best < 0) break
      w.splice(bestPair, 2, w[bestPair] + w[bestPair + 1])
    }
    return w
  }
  function tokenize(text) {
    const ids = []
    let rem = text
    while (rem.length > 0) {
      let hit = false
      for (const p of addedList) {
        if (rem.startsWith(p)) { ids.push(addedMap[p]); rem = rem.slice(p.length); hit = true; break }
      }
      if (hit) continue
      const m = rem.match(/^(\s*\S+|\s+)/)
      if (!m) break
      const chunk = m[0]
      rem = rem.slice(chunk.length)
      for (const t of bpe([...Array.from(new TextEncoder().encode(chunk)).map(b => byteEnc[b]).join('')])) ids.push(vocab[t] ?? 0)
    }
    return ids
  }
  function decode(ids) {
    const chars = ids.map(id => idToToken[id] ?? '').join('').split('')
    return new TextDecoder().decode(new Uint8Array(chars.map(c => byteDec[c] ?? c.charCodeAt(0))))
  }
  return { tokenize, decode }
}
