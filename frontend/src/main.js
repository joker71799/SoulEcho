// SoulEcho 轻量化前端对接逻辑
// 后端接口（经 Vite 代理转发到 127.0.0.1:9000）：
//   POST /api/v1/chat  body: {user_id, conversation_id, content} -> {reply}
//   GET  /health       -> {status, project}

// ---------- ID 管理 ----------
// user_id：长期记忆隔离，跨会话稳定，存 localStorage
// conversation_id：单次会话多轮上下文（thread_id），存 sessionStorage，可“新会话”重置
function genId(prefix) {
  const rand = Math.random().toString(36).slice(2, 10);
  return `${prefix}_${Date.now().toString(36)}_${rand}`;
}

function getUserId() {
  let id = localStorage.getItem('soulecho_user_id');
  if (!id) {
    id = genId('user');
    localStorage.setItem('soulecho_user_id', id);
  }
  return id;
}

function getConversationId() {
  let id = sessionStorage.getItem('soulecho_conversation_id');
  if (!id) {
    id = genId('conv');
    sessionStorage.setItem('soulecho_conversation_id', id);
  }
  return id;
}

const USER_ID = getUserId();
let conversationId = getConversationId();

// ---------- DOM ----------
const messagesEl = document.getElementById('messages');
const inputEl = document.getElementById('input');
const sendBtn = document.getElementById('send');
const newChatBtn = document.getElementById('new-chat');
const healthEl = document.getElementById('health');

// ---------- 渲染 ----------
function addMessage(role, text, extraClass = '') {
  const wrap = document.createElement('div');
  wrap.className = `msg ${role}`;

  const avatar = document.createElement('div');
  avatar.className = 'avatar';
  avatar.textContent = role === 'user' ? '🙋' : '🌿';

  const bubble = document.createElement('div');
  bubble.className = `bubble ${extraClass}`.trim();
  bubble.textContent = text;

  wrap.appendChild(avatar);
  wrap.appendChild(bubble);
  messagesEl.appendChild(wrap);
  messagesEl.scrollTop = messagesEl.scrollHeight;
  return bubble;
}

// ---------- 健康检查 ----------
async function checkHealth() {
  healthEl.className = 'health';
  healthEl.textContent = '● 检测中';
  try {
    const res = await fetch('/health');
    if (res.ok) {
      healthEl.className = 'health ok';
      healthEl.textContent = '● 服务在线';
    } else {
      throw new Error(String(res.status));
    }
  } catch {
    healthEl.className = 'health bad';
    healthEl.textContent = '● 服务离线';
  }
}

// ---------- 发送 ----------
let sending = false;

async function sendMessage() {
  const content = inputEl.value.trim();
  if (!content || sending) return;

  sending = true;
  sendBtn.disabled = true;
  inputEl.value = '';

  addMessage('user', content);
  const pending = addMessage('agent', '正在思考…', 'typing');

  try {
    const res = await fetch('/api/v1/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        user_id: USER_ID,
        conversation_id: conversationId,
        content,
      }),
    });

    if (!res.ok) {
      const detail = await res.text().catch(() => '');
      throw new Error(`HTTP ${res.status} ${detail}`.trim());
    }

    const data = await res.json();
    pending.className = 'bubble';
    pending.textContent = data.reply ?? '(空回复)';
  } catch (err) {
    pending.className = 'bubble error';
    pending.textContent = `请求失败：${err.message}。请确认后端已启动（127.0.0.1:9000）。`;
    checkHealth();
  } finally {
    sending = false;
    sendBtn.disabled = false;
    messagesEl.scrollTop = messagesEl.scrollHeight;
    inputEl.focus();
  }
}

// ---------- 新会话 ----------
function startNewConversation() {
  conversationId = genId('conv');
  sessionStorage.setItem('soulecho_conversation_id', conversationId);
  messagesEl.innerHTML = '';
  addMessage(
    'agent',
    '已开启新的会话。无论是开心还是难过，都可以在这里说说。'
  );
}

// ---------- 事件绑定 ----------
sendBtn.addEventListener('click', sendMessage);
newChatBtn.addEventListener('click', startNewConversation);
inputEl.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
});

// ---------- 初始化 ----------
checkHealth();
inputEl.focus();
