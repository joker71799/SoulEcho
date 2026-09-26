// SoulEcho 轻量化前端对接逻辑
// 后端接口（经 Vite 代理转发到 127.0.0.1:9000）：
//   POST /api/v1/chat    body: {user_id, conversation_id, content} -> {status, reply?, steer?}
//   POST /api/v1/resume  body: {user_id, conversation_id, support_mode} -> {status, reply?, steer?}
//   GET  /health         -> {status, project}
// status："done" 直接渲染回复；"direction_required" 渲染方向引导一键选项（场景B HITL）。

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

// 切换用户身份：写入 localStorage 并同步输入框显示
function setUserId(id) {
  userId = id;
  localStorage.setItem('soulecho_user_id', id);
  if (userIdInputEl) userIdInputEl.value = id;
}

function getConversationId() {
  let id = sessionStorage.getItem('soulecho_conversation_id');
  if (!id) {
    id = genId('conv');
    sessionStorage.setItem('soulecho_conversation_id', id);
  }
  return id;
}

let userId = getUserId();
let conversationId = getConversationId();

// ---------- DOM ----------
const messagesEl = document.getElementById('messages');
const inputEl = document.getElementById('input');
const sendBtn = document.getElementById('send');
const newChatBtn = document.getElementById('new-chat');
const healthEl = document.getElementById('health');
const userIdInputEl = document.getElementById('user-id-input');
const switchUserBtn = document.getElementById('switch-user');
const randomUserBtn = document.getElementById('random-user');

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

// ---------- 发送 / 恢复 ----------
let sending = false;
// 方向引导待点选时，暂停普通输入发送，避免在“已中断的 thread”上重复 invoke 造成异常
let awaitingDirection = false;

// 统一处理后端返回：done 渲染回复（危机时附带热线卡）；direction_required 渲染一键选项
function renderResult(data, pending) {
  clearSteer();
  if (data.status === 'direction_required') {
    pending.className = 'bubble';
    pending.textContent = data.steer?.question || '先看看这一刻你更需要我怎么陪你？';
    awaitingDirection = true;
    renderSteerOptions(data.steer?.options || []);
  } else {
    pending.className = 'bubble';
    pending.textContent = data.reply ?? '(空回复)';
    awaitingDirection = false;
    // 危机命中：后端随 reply 一起返回 crisis 卡片，额外渲染一张醒目的求助卡
    if (data.crisis) renderCrisisCard(data.crisis);
  }
}

// 危机热线卡片：区别于普通气泡，醒目呈现可直拨的求助电话
function renderCrisisCard(crisis) {
  const wrap = document.createElement('div');
  wrap.className = 'msg agent crisis';

  const card = document.createElement('div');
  card.className = 'crisis-card';

  const title = document.createElement('div');
  title.className = 'crisis-title';
  title.textContent = crisis.title || '你并不孤单，请让这些专业的人帮帮你';
  card.appendChild(title);

  if (crisis.message) {
    const msg = document.createElement('div');
    msg.className = 'crisis-msg';
    msg.textContent = crisis.message;
    card.appendChild(msg);
  }

  // 紧急电话（imminent 时引导优先拨打）与常规热线统一渲染为可点击 tel: 链接
  [...(crisis.emergency || []), ...(crisis.hotlines || [])].forEach((h) => {
    card.appendChild(crisisLine(h));
  });

  wrap.appendChild(card);
  messagesEl.appendChild(wrap);
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

function crisisLine(h) {
  const row = document.createElement('a');
  row.className = 'crisis-line';
  row.href = `tel:${String(h.number || '').replace(/[^0-9+]/g, '')}`;
  row.textContent = `${h.name}：${h.number}${h.desc ? '（' + h.desc + '）' : ''}`;
  return row;
}

// chat 与 resume 共用：发起请求 -> 渲染结果 -> 复位发送态
async function runRequest(url, body, pending) {
  try {
    const res = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });

    if (!res.ok) {
      const detail = await res.text().catch(() => '');
      throw new Error(`HTTP ${res.status} ${detail}`.trim());
    }

    const data = await res.json();
    renderResult(data, pending);
  } catch (err) {
    pending.className = 'bubble error';
    pending.textContent = `请求失败：${err.message}。请确认后端已启动（127.0.0.1:9000）。`;
    awaitingDirection = false;
    checkHealth();
  } finally {
    sending = false;
    sendBtn.disabled = false;
    messagesEl.scrollTop = messagesEl.scrollHeight;
    inputEl.focus();
  }
}

async function sendMessage() {
  const content = inputEl.value.trim();
  if (!content || sending || awaitingDirection) return;

  sending = true;
  sendBtn.disabled = true;
  inputEl.value = '';

  addMessage('user', content);
  const pending = addMessage('agent', '正在思考…', 'typing');

  await runRequest(
    '/api/v1/chat',
    { user_id: userId, conversation_id: conversationId, content },
    pending
  );
}

// 用户点选陪伴方向：移除选项、以轻量方式回显，再调 /api/v1/resume 恢复生成
async function submitDirection(value, label, chipWrap) {
  if (sending) return;
  sending = true;
  sendBtn.disabled = true;
  awaitingDirection = false;
  if (chipWrap) chipWrap.remove();
  if (label) addMessage('user', label);
  const pending = addMessage('agent', '正在思考…', 'typing');

  await runRequest(
    '/api/v1/resume',
    { user_id: userId, conversation_id: conversationId, support_mode: value },
    pending
  );
}

// ---------- 方向引导选项渲染 ----------
let steerEl = null;
function clearSteer() {
  if (steerEl) {
    steerEl.remove();
    steerEl = null;
  }
}
function renderSteerOptions(options) {
  clearSteer();
  steerEl = document.createElement('div');
  steerEl.className = 'msg agent steer';
  const box = document.createElement('div');
  box.className = 'steer-options';
  options.forEach((opt) => {
    // CF 推断命中的选项带 recommended=true：高亮 + “AI 推荐”徽标，让用户对着问句一眼找到对应项、可一键采纳
    const recommended = opt.recommended === true;
    const btn = document.createElement('button');
    btn.className = recommended ? 'steer-chip recommended' : 'steer-chip';
    btn.textContent = opt.label;
    if (recommended) {
      const tag = document.createElement('span');
      tag.className = 'steer-tag';
      tag.textContent = 'AI 推荐';
      btn.appendChild(tag);
    }
    const tips = [];
    if (opt.hint) tips.push(opt.hint);
    if (recommended) tips.push('这是我听下来最贴合的方向，可直接采纳');
    if (tips.length) btn.title = tips.join(' · ');
    btn.addEventListener('click', () => submitDirection(opt.value, opt.label, steerEl));
    box.appendChild(btn);
  });
  steerEl.appendChild(box);
  messagesEl.appendChild(steerEl);
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

// ---------- 新会话 ----------
function startNewConversation() {
  clearSteer();
  awaitingDirection = false;
  conversationId = genId('conv');
  sessionStorage.setItem('soulecho_conversation_id', conversationId);
  messagesEl.innerHTML = '';
  addMessage(
    'agent',
    '已开启新的会话。无论是开心还是难过，都可以在这里说说。'
  );
}

// ---------- 切换用户身份（测试用） ----------
// 切换后重置会话，避免不同身份的长期记忆 / 多轮上下文混用
function switchUserId(newId) {
  const id = String(newId || '').trim();
  if (!id || id === userId) {
    userIdInputEl.value = userId;
    return;
  }
  setUserId(id);
  startNewConversation();
  addMessage('agent', `已切换到用户：${id}`);
}

// ---------- 事件绑定 ----------
sendBtn.addEventListener('click', sendMessage);
newChatBtn.addEventListener('click', startNewConversation);
switchUserBtn.addEventListener('click', () => switchUserId(userIdInputEl.value));
randomUserBtn.addEventListener('click', () => switchUserId(genId('user')));
userIdInputEl.addEventListener('keydown', (e) => {
  if (e.key === 'Enter') {
    e.preventDefault();
    switchUserId(userIdInputEl.value);
  }
});
inputEl.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
});

// ---------- 初始化 ----------
userIdInputEl.value = userId;
checkHealth();
inputEl.focus();
