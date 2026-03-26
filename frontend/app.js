/* app.js — HR Avatar Frontend Logic */

const API    = 'http://localhost:8000';
const SECRET = 'dev-secret';   // matches LMS_SHARED_SECRET in config.py

// ── State ─────────────────────────────────────────────────────────────────────
let sessionId     = null;
let isProcessing  = false;
// ── VAD State ──────────────────────────────────────────────────────────────
let vadActive      = false;   // VAD enabled (mic button toggles)
let isSpeaking     = false;   // currently detected speech
let silenceFrames  = 0;
let speechFrames   = 0;
let audioContext   = null;
let analyser       = null;
let micStream      = null;
let vadRecorder    = null;
let vadChunks      = [];

const SPEECH_THRESHOLD = 0.018;   // RMS amplitude — analogous to Silero's 0.5 prob
const SILENCE_LIMIT    = 42;      // frames @ ~60fps ≈ 700 ms (mirrors Python SILENCE_LIMIT)
const SPEECH_CONFIRM   = 3;       // frames needed to confirm speech start

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
const avatarStatus  = document.getElementById('avatar-status');
const playOverlay   = document.getElementById('play-overlay');
const playBtn       = document.getElementById('play-btn');

function setAvatar(state) {
  avatarContainer.className = `avatar-container ${state}`;
  avatarStatus.textContent  = state === 'thinking' ? 'Thinking…' : '';
  if (state !== 'speaking') playOverlay.classList.add('hidden');
}

function playAvatarVideo(url) {
  setAvatar('speaking');
  playOverlay.classList.add('hidden');

  // Clear previous handlers before loading new src
  avatarVideo.oncanplay = null;
  avatarVideo.onerror   = null;
  avatarVideo.onended   = null;

  avatarVideo.src = url;
  avatarVideo.load();

  avatarVideo.onerror = () => setAvatar('idle');
  avatarVideo.onended = () => setAvatar('idle');

  // Wait until enough data is buffered before playing
  avatarVideo.oncanplay = () => {
    avatarVideo.play().catch(() => {
      // Browser blocked autoplay (unmuted video policy) — show click overlay
      playOverlay.classList.remove('hidden');
    });
  };
}

// Click-to-play when autoplay is blocked
playBtn.addEventListener('click', () => {
  playOverlay.classList.add('hidden');
  avatarVideo.play().catch(() => {});
});

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

  if (vadActive) {
    if (val) {
      // Hide VAD bar while backend is processing (thinking bubble takes over)
      recordingBar.classList.add('hidden');
    } else if (!isSpeaking) {
      // Restore "listening" indicator once processing completes
      updateVadUI('listening');
    }
  }
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

    // Trigger personalised spoken welcome — calls backend TTS+lipsync, no LLM
    fetchWelcome(sessionId);

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

// ── Welcome greeting (called once after login) ────────────────────────────────
async function fetchWelcome(sid) {
  setProcessing(true);
  showThinking();

  try {
    const res = await fetch(`${API}/session/welcome`, {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({ session_id: sid }),
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
    // Welcome failure is non-fatal — just show idle avatar, user can still chat
    removeThinking();
    setAvatar('idle');
  } finally {
    setProcessing(false);
    // Auto-activate VAD after welcome so user can speak immediately
    if (!vadActive) startVAD();
  }
}

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

// ── Voice / VAD ───────────────────────────────────────────────────────────────
// Mic button toggles VAD on/off (mute/unmute). VAD auto-starts after welcome.
micBtn.addEventListener('click', () => {
  if (!sessionId) return;
  if (vadActive) {
    stopVAD();
  } else {
    startVAD();
  }
});

async function startVAD() {
  if (vadActive) return;
  try {
    micStream = await navigator.mediaDevices.getUserMedia({ audio: true });
    audioContext = new AudioContext();
    const source = audioContext.createMediaStreamSource(micStream);
    analyser = audioContext.createAnalyser();
    analyser.fftSize = 1024;
    source.connect(analyser);

    vadActive     = true;
    isSpeaking    = false;
    silenceFrames = 0;
    speechFrames  = 0;

    updateVadUI('listening');
    monitorAudio();
  } catch (err) {
    showError('Microphone access denied. Please allow microphone access in your browser.');
  }
}

function stopVAD() {
  vadActive = false;
  if (vadRecorder && vadRecorder.state !== 'inactive') vadRecorder.stop();
  if (micStream)      { micStream.getTracks().forEach(t => t.stop()); micStream = null; }
  if (audioContext)   { audioContext.close(); audioContext = null; }
  analyser   = null;
  isSpeaking = false;
  vadRecorder = null;
  vadChunks   = [];
  updateVadUI('off');
}

function getRMS(data) {
  let sum = 0;
  for (let i = 0; i < data.length; i++) {
    const v = (data[i] / 128.0) - 1.0;   // normalise byte → -1..+1
    sum += v * v;
  }
  return Math.sqrt(sum / data.length);
}

function monitorAudio() {
  if (!vadActive || !analyser) return;
  const data = new Uint8Array(analyser.fftSize);

  function loop() {
    if (!vadActive || !analyser) return;
    analyser.getByteTimeDomainData(data);
    const rms = getRMS(data);

    if (!isProcessing) {
      if (!isSpeaking) {
        // Waiting for speech to begin
        if (rms > SPEECH_THRESHOLD) {
          speechFrames++;
          if (speechFrames >= SPEECH_CONFIRM) onSpeechStart();
        } else {
          speechFrames = 0;
        }
      } else {
        // Currently speaking — watch for silence
        if (rms < SPEECH_THRESHOLD) {
          silenceFrames++;
          if (silenceFrames >= SILENCE_LIMIT) onSpeechEnd();
        } else {
          silenceFrames = 0;
        }
      }
    }

    requestAnimationFrame(loop);
  }

  requestAnimationFrame(loop);
}

function onSpeechStart() {
  if (isSpeaking || !micStream) return;
  isSpeaking    = true;
  silenceFrames = 0;
  speechFrames  = 0;
  updateVadUI('speaking');

  vadChunks = [];
  const mimeType = MediaRecorder.isTypeSupported('audio/webm;codecs=opus')
    ? 'audio/webm;codecs=opus' : 'audio/webm';

  vadRecorder = new MediaRecorder(micStream, { mimeType });
  vadRecorder.ondataavailable = (e) => { if (e.data.size > 0) vadChunks.push(e.data); };
  vadRecorder.onstop = () => {
    if (vadChunks.length > 0) {
      const blob = new Blob(vadChunks, { type: mimeType });
      sendAudio(blob);
    }
    vadChunks = [];
  };
  vadRecorder.start(100);
}

function onSpeechEnd() {
  if (!isSpeaking) return;
  isSpeaking    = false;
  silenceFrames = 0;
  speechFrames  = 0;
  if (vadRecorder && vadRecorder.state === 'recording') vadRecorder.stop();
  // UI reverts to 'listening' via setProcessing(false) once backend responds
}

function updateVadUI(state) {
  // state: 'off' | 'listening' | 'speaking'
  micBtn.classList.remove('vad-off', 'vad-listening', 'vad-speaking');
  const statusText = document.getElementById('rec-status-text');

  if (state === 'listening') {
    micBtn.classList.add('vad-listening');
    micBtn.title = 'VAD active — listening (click to mute)';
    if (statusText) statusText.textContent = 'Listening… speak to send a voice message';
    recordingBar.classList.add('listening');
    recordingBar.classList.remove('hidden');
  } else if (state === 'speaking') {
    micBtn.classList.add('vad-speaking');
    micBtn.title = 'Speech detected — recording';
    if (statusText) statusText.textContent = 'Speech detected — recording…';
    recordingBar.classList.remove('listening');
    recordingBar.classList.remove('hidden');
  } else {
    micBtn.classList.add('vad-off');
    micBtn.title = 'Click to activate voice detection';
    recordingBar.classList.remove('listening');
    recordingBar.classList.add('hidden');
  }
}

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

  if (vadActive) stopVAD();
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
