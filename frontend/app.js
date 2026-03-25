/* app.js — HR Avatar Frontend Logic */

const API    = 'http://localhost:8000';
const SECRET = 'dev-secret';   // matches LMS_SHARED_SECRET in config.py

// ── State ─────────────────────────────────────────────────────────────────────
let sessionId     = null;
let isProcessing  = false;
let mediaRecorder = null;
let audioChunks   = [];

// ── DOM refs ──────────────────────────────────────────────────────────────────
const loginScreen      = document.getElementById('login-screen');
const chatScreen       = document.getElementById('chat-screen');
const loginForm        = document.getElementById('login-form');
const messagesEl       = document.getElementById('messages');
const msgInput         = document.getElementById('msg-input');
const sendBtn          = document.getElementById('send-btn');
const micBtn           = document.getElementById('mic-btn');
const avatarContainer  = document.getElementById('avatar-container');
const avatarVideo      = document.getElementById('avatar-video');
const recordingBar     = document.getElementById('recording-bar');

// ── Helpers ───────────────────────────────────────────────────────────────────
const profile = () => ({
  user_id:          document.getElementById('inp-userid').value.trim() || 'emp_001',
  name:             document.getElementById('inp-name').value.trim(),
  job_role:         document.getElementById('inp-role').value.trim(),
  department:       document.getElementById('inp-dept').value.trim(),
  skill_level:      document.getElementById('inp-skill').value,
  known_skills:     document.getElementById('inp-skills').value
                      .split(',').map(s => s.trim()).filter(Boolean),
  enrolled_courses: [],
  context:          'avatar_chat',
});

function escapeHtml(str) {
  return str
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/\n/g, '<br>');
}

// ── Avatar states: idle | thinking | speaking ─────────────────────────────────
const avatarStatus = document.getElementById('avatar-status');

function setAvatar(state) {
  avatarContainer.className = `avatar-container ${state}`;
  avatarStatus.textContent = state === 'thinking' ? 'Thinking…' : '';
}

function playAvatarVideo(url) {
  setAvatar('speaking');
  avatarVideo.src = url;
  avatarVideo.play().catch(() => {});
  avatarVideo.onended = () => setAvatar('idle');
  avatarVideo.onerror = () => setAvatar('idle');
}

// ── Messages ──────────────────────────────────────────────────────────────────
function addMessage(role, html) {
  const wrap = document.createElement('div');
  wrap.className = `message ${role}`;

  if (role === 'user') {
    wrap.innerHTML = `<div class="bubble">${html}</div>`;
  } else if (role === 'assistant') {
    wrap.innerHTML = `<div class="msg-avatar">HR</div><div class="bubble">${html}</div>`;
  } else {
    wrap.innerHTML = `<div class="bubble">${html}</div>`;
  }

  messagesEl.appendChild(wrap);
  messagesEl.scrollTop = messagesEl.scrollHeight;
  return wrap;
}

function showThinking() {
  setAvatar('thinking');
  const wrap = document.createElement('div');
  wrap.className = 'message assistant';
  wrap.id = 'thinking-bubble';
  wrap.innerHTML = `
    <div class="msg-avatar">HR</div>
    <div class="bubble thinking-bubble">
      <span></span><span></span><span></span>
    </div>`;
  messagesEl.appendChild(wrap);
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

function removeThinking() {
  document.getElementById('thinking-bubble')?.remove();
}

function setProcessing(val) {
  isProcessing  = val;
  sendBtn.disabled  = val;
  msgInput.disabled = val;
  document.querySelectorAll('.quick-btn').forEach(b => b.disabled = val);
}

// ── Session Start ─────────────────────────────────────────────────────────────
loginForm.addEventListener('submit', async (e) => {
  e.preventDefault();
  const btn   = document.getElementById('start-btn');
  const label = document.getElementById('start-btn-label');
  const spin  = document.getElementById('btn-spinner');

  btn.disabled = true;
  label.textContent = 'Connecting…';
  spin.classList.remove('hidden');

  try {
    const p = profile();
    const res = await fetch(`${API}/session/start`, {
      method:  'POST',
      headers: {
        'Content-Type':  'application/json',
        'Authorization': `Bearer ${SECRET}`,
      },
      body: JSON.stringify(p),
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `HTTP ${res.status}`);
    }

    const data = await res.json();
    sessionId = data.session_id;

    // Populate employee card & header
    document.getElementById('emp-name').textContent  = p.name;
    document.getElementById('emp-role').textContent  = `${p.job_role} · ${p.department}`;
    document.getElementById('header-sub').textContent = `Chatting as ${p.name}`;

    const tagsEl = document.getElementById('emp-tags');
    tagsEl.innerHTML = p.known_skills.slice(0, 3)
      .map(s => `<span class="tag">${escapeHtml(s)}</span>`)
      .join('');

    // Switch screens
    loginScreen.classList.remove('active');
    chatScreen.classList.add('active');

    // Greeting (no API call needed — frontend-generated)
    addMessage('assistant',
      `Hello <strong>${escapeHtml(p.name)}</strong>! I'm your HR Avatar. ` +
      `I can see you're a <strong>${escapeHtml(p.job_role)}</strong> in ` +
      `<strong>${escapeHtml(p.department)}</strong> with ` +
      `<strong>${p.skill_level.toLowerCase()}</strong>-level skills. ` +
      `How can I help you today?`
    );

  } catch (err) {
    spin.classList.add('hidden');
    label.textContent = 'Start Conversation';
    btn.disabled = false;
    showError(
      `Could not connect to the HR Avatar backend.\n\n` +
      `Make sure the server is running:\n  uvicorn web.app:app --host 0.0.0.0 --port 8000\n\n` +
      `Error: ${err.message}`
    );
  }
});

// ── Text Chat ─────────────────────────────────────────────────────────────────
sendBtn.addEventListener('click', submitText);

msgInput.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    submitText();
  }
});

// Auto-resize textarea
msgInput.addEventListener('input', () => {
  msgInput.style.height = 'auto';
  msgInput.style.height = Math.min(msgInput.scrollHeight, 120) + 'px';
});

function submitText() {
  const text = msgInput.value.trim();
  if (!text || isProcessing || !sessionId) return;
  msgInput.value = '';
  msgInput.style.height = 'auto';
  addMessage('user', escapeHtml(text));
  callChat(text);
}

async function callChat(text) {
  setProcessing(true);
  showThinking();

  try {
    const res = await fetch(`${API}/chat`, {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({ session_id: sessionId, message: text }),
    });

    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();

    removeThinking();
    addMessage('assistant', escapeHtml(data.reply));

    if (data.video_url) {
      playAvatarVideo(`${API}${data.video_url}`);
    } else {
      setAvatar('idle');
    }

  } catch (err) {
    removeThinking();
    addMessage('error', `⚠️ ${escapeHtml(err.message)}`);
    setAvatar('idle');
  } finally {
    setProcessing(false);
  }
}

// ── Quick questions ───────────────────────────────────────────────────────────
document.querySelectorAll('.quick-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    if (isProcessing || !sessionId) return;
    const text = btn.dataset.q;
    addMessage('user', escapeHtml(text));
    callChat(text);
  });
});

// ── Voice Recording ───────────────────────────────────────────────────────────
micBtn.addEventListener('click', async () => {
  // Stop if already recording
  if (mediaRecorder && mediaRecorder.state === 'recording') {
    mediaRecorder.stop();
    return;
  }

  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    audioChunks = [];

    // Prefer wav-compatible mime type
    const mimeType = MediaRecorder.isTypeSupported('audio/webm;codecs=opus')
      ? 'audio/webm;codecs=opus'
      : 'audio/webm';

    mediaRecorder = new MediaRecorder(stream, { mimeType });
    mediaRecorder.ondataavailable = (e) => { if (e.data.size > 0) audioChunks.push(e.data); };

    mediaRecorder.onstop = async () => {
      stream.getTracks().forEach(t => t.stop());
      micBtn.classList.remove('recording');
      recordingBar.classList.add('hidden');

      const blob = new Blob(audioChunks, { type: mimeType });
      await sendAudio(blob);
    };

    mediaRecorder.start(100);
    micBtn.classList.add('recording');
    recordingBar.classList.remove('hidden');

  } catch (err) {
    showError('Microphone access denied. Please allow microphone access in your browser to use voice input.');
  }
});

async function sendAudio(blob) {
  setProcessing(true);
  showThinking();

  try {
    const form = new FormData();
    form.append('session_id', sessionId);        // sent as a Form field — matches Form(...) on backend
    form.append('audio', blob, 'recording.webm');

    const res = await fetch(`${API}/chat/audio`, {
      method: 'POST',
      body:   form,
    });

    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();

    removeThinking();
    addMessage('user', '🎤 <em>(voice message)</em>');
    addMessage('assistant', escapeHtml(data.reply));

    if (data.video_url) {
      playAvatarVideo(`${API}${data.video_url}`);
    } else {
      setAvatar('idle');
    }

  } catch (err) {
    removeThinking();
    addMessage('error', `⚠️ Audio error: ${escapeHtml(err.message)}`);
    setAvatar('idle');
  } finally {
    setProcessing(false);
  }
}

// ── End Session ───────────────────────────────────────────────────────────────
document.getElementById('end-btn').addEventListener('click', async () => {
  if (!sessionId) return;
  if (!confirm('End this session and return to the login screen?')) return;

  await fetch(`${API}/session/${sessionId}`, { method: 'DELETE' }).catch(() => {});
  sessionId = null;

  // Reset UI
  chatScreen.classList.remove('active');
  loginScreen.classList.add('active');
  messagesEl.innerHTML = '';
  setAvatar('idle');

  const btn   = document.getElementById('start-btn');
  const label = document.getElementById('start-btn-label');
  const spin  = document.getElementById('btn-spinner');
  btn.disabled = false;
  label.textContent = 'Start Conversation';
  spin.classList.add('hidden');
});

// ── Error dialog ──────────────────────────────────────────────────────────────
function showError(msg) {
  alert(msg);
}
