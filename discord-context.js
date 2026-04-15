const contextStore = new Map();

function getKey(guildId, channelId) {
  return `${guildId}:${channelId}`;
}

function addMessage(guildId, channelId, userId, role, text) {
  if (!guildId || !channelId || !userId || !role || !text) {
    throw new Error('addMessage requires guildId, channelId, userId, role, text');
  }

  const key = getKey(guildId, channelId);
  if (!contextStore.has(key)) {
    contextStore.set(key, []);
  }

  const messages = contextStore.get(key);
  messages.push({
    userId,
    role,
    text,
    timestamp: Date.now(),
  });

  if (messages.length > 50) {
    messages.shift();
  }
}

function getContext(guildId, channelId) {
  const key = getKey(guildId, channelId);
  if (!contextStore.has(key)) {
    return [];
  }

  const messages = contextStore.get(key);
  const start = Math.max(0, messages.length - 20);
  return messages.slice(start);
}

function clearContext(guildId, channelId) {
  const key = getKey(guildId, channelId);
  contextStore.delete(key);
}

export { addMessage, getContext, clearContext };
