/* app.js — HR Avatar Frontend Logic */

const API    = 'http://localhost:8000';
const SECRET = 'dev-secret';

// ── State ─────────────────────────────────────────────────────────────────────
let sessionId    = null;
let isProcessing = false;

// VAD state
let vadInstance = null;

// ── DOM refs ──────────────────────────────────────────────────────────────────
const loginScreen     = document.getElementById('login-screen');
const chatScreen      = document.getElementById('chat-screen');
const loginForm       = document.getElementById('login-form');
const messagesEl      = document.getElementById('messages');
const msgInput        = document.getElementById('msg-input');
const sendBtn         = document.getElementById('send-btn');
const micBtn          = document.getElementById('mic-btn');
const avatarContainer = document.getElementById('avatar-container');
const avatarVideo     = document.getElementById('avatar-video');
const avatarIdle      = document.getElementById('avatar-idle');
const recordingBar    = document.getElementById('recording-bar');
const avatarStatus    = document.getElementById('avatar-status');
const playOverlay     = document.getElementById('play-overlay');
const playBtn         = document.getElementById('play-btn');
const micLevelBar     = document.getElementById('mic-level-bar');

// ── Avatar states ─────────────────────────────────────────────────────────────
function setAvatar(state) {
  avatarContainer.className = `avatar-container ${state}`;
  avatarStatus.textContent  = state === 'thinking' ? 'Thinking…'
                            : state === 'waiting'  ? 'Preparing…'
                            : '';
  if (state !== 'speaking') {
    playOverlay.classList.add('hidden');
    ensureIdleVideoPlaying();
  }
}

function ensureIdleVideoPlaying() {
  if (!avatarIdle) return;
  const tryPlay = () => avatarIdle.play().catch(() => {});
  if (avatarIdle.readyState >= 2) {
    if (avatarIdle.paused) tryPlay();
  } else {
    avatarIdle.addEventListener('canplay', tryPlay, { once: true });
    avatarIdle.load();
  }
}

playBtn.addEventListener('click', () => {
  playOverlay.classList.add('hidden');
  avatarVideo.play().catch(() => {});
});

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
    .replace(/&/g, '&amp;').replace(/</g, '&lt;')
    .replace(/>/g, '&gt;').replace(/\n/g, '<br>');
}

// Render assistant message text as safe HTML with clickable course links.
// Handles markdown links [Title](url) and converts newlines to <br>.
function formatMessage(text) {
  const escaped = text
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');
  return escaped
    .replace(
      /\[([^\]]+)\]\((https?:\/\/[^)]+)\)/g,
      '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>'
    )
    .replace(/\n/g, '<br>');
}

// ── WAV encoder — converts Silero VAD Float32 output to a WAV blob ───────────
function float32ToWav(samples, sampleRate) {
  const buf  = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(buf);
  const str  = (off, s) => [...s].forEach((c, i) => view.setUint8(off + i, c.charCodeAt(0)));
  str(0, 'RIFF');  view.setUint32(4, 36 + samples.length * 2, true);
  str(8, 'WAVE');  str(12, 'fmt ');
  view.setUint32(16, 16, true);  view.setUint16(20, 1, true);   // PCM
  view.setUint16(22, 1, true);   view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);   view.setUint16(34, 16, true);
  str(36, 'data'); view.setUint32(40, samples.length * 2, true);
  let off = 44;
  for (let i = 0; i < samples.length; i++, off += 2) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(off, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
  }
  return new Blob([buf], { type: 'audio/wav' });
}

// ── Messages ──────────────────────────────────────────────────────────────────
function addMessage(role, html) {
  const wrap = document.createElement('div');
  wrap.className = `message ${role}`;
  if (role === 'assistant') {
    wrap.innerHTML = `<div class="msg-avatar">HR</div><div class="bubble">${html}</div>`;
  } else {
    wrap.innerHTML = `<div class="bubble">${html}</div>`;
  }
  messagesEl.appendChild(wrap);
  messagesEl.scrollTop = messagesEl.scrollHeight;
  return wrap;
}

function createAssistantBubble() {
  const wrap = document.createElement('div');
  wrap.className = 'message assistant';
  wrap.innerHTML = '<div class="msg-avatar">HR</div><div class="bubble"></div>';
  messagesEl.appendChild(wrap);
  messagesEl.scrollTop = messagesEl.scrollHeight;
  return wrap.querySelector('.bubble');
}

function streamTextIntoBubble(text, bubble, videoDurationSec) {
  // Stream a URL-stripped version in sync with the TTS audio, then replace
  // the bubble content with the fully formatted version (clickable links).
  const streamText = text.replace(/\[([^\]]*)\]\([^)]*\)/g, '$1')
                         .replace(/https?:\/\/\S+/g, '')
                         .replace(/\s{2,}/g, ' ').trim();
  const words = streamText.split(' ').filter(Boolean);
  if (!words.length) { bubble.innerHTML = formatMessage(text); return; }
  const msPerWord = Math.max(120, (videoDurationSec * 1000) / words.length);
  let i = 0;
  const tick = () => {
    if (i >= words.length) {
      bubble.innerHTML = formatMessage(text);
      messagesEl.scrollTop = messagesEl.scrollHeight;
      return;
    }
    bubble.textContent += (i === 0 ? '' : ' ') + words[i++];
    messagesEl.scrollTop = messagesEl.scrollHeight;
    setTimeout(tick, msPerWord);
  };
  tick();
}

function showThinking() {
  setAvatar('thinking');
  const wrap = document.createElement('div');
  wrap.className = 'message assistant';
  wrap.id = 'thinking-bubble';
  wrap.innerHTML = `
    <div class="msg-avatar">HR</div>
    <div class="bubble thinking-bubble"><span></span><span></span><span></span></div>`;
  messagesEl.appendChild(wrap);
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

function removeThinking() {
  document.getElementById('thinking-bubble')?.remove();
}

// ── Processing state ──────────────────────────────────────────────────────────
function setProcessing(val) {
  isProcessing    = val;
  sendBtn.disabled  = val;
  msgInput.disabled = val;
  micBtn.disabled   = val;
  document.querySelectorAll('.quick-btn').forEach(b => b.disabled = val);
}

// ── Video + text playback ─────────────────────────────────────────────────────
function playVideoAndStreamText(videoUrl, replyText) {
  setAvatar('speaking');
  avatarVideo.oncanplay = null;
  avatarVideo.onerror   = null;
  avatarVideo.onended   = null;

  avatarVideo.src = videoUrl.startsWith('http') ? videoUrl : `${API}${videoUrl}`;
  avatarVideo.load();

  avatarVideo.onerror = () => {
    console.error('[Video] playback error — code:', avatarVideo.error?.code, avatarVideo.src);
    setAvatar('idle');
    addMessage('assistant', formatMessage(replyText));
  };

  avatarVideo.onended = () => {
    setAvatar('idle');
    ensureIdleVideoPlaying();
  };

  avatarVideo.oncanplay = () => {
    avatarVideo.play().catch(() => playOverlay.classList.remove('hidden'));
    const dur = isFinite(avatarVideo.duration) && avatarVideo.duration > 0
      ? avatarVideo.duration : 8;
    const bubble = createAssistantBubble();
    streamTextIntoBubble(replyText, bubble, dur);
  };
}

async function waitForVideoThenPlay(jobId, replyText) {
  const maxMs = 180000;
  const start  = Date.now();
  while (Date.now() - start < maxMs) {
    await new Promise(r => setTimeout(r, 250));
    try {
      const res  = await fetch(`${API}/video/status/${jobId}`);
      if (!res.ok) break;
      const data = await res.json();
      if (data.ready && data.video_url) {
        playVideoAndStreamText(data.video_url, replyText);
        return;
      }
      if (data.error) { console.error('[Video] lipsync job failed:', data.error); break; }
    } catch (e) { break; }
  }
  addMessage('assistant', formatMessage(replyText));
  setAvatar('idle');
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
    const p   = profile();
    const res = await fetch(`${API}/session/start`, {
      method:  'POST',
      headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${SECRET}` },
      body:    JSON.stringify(p),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `HTTP ${res.status}`);
    }

    const data = await res.json();
    sessionId = data.session_id;

    document.getElementById('emp-name').textContent  = p.name;
    document.getElementById('emp-role').textContent  = `${p.job_role} · ${p.department}`;
    document.getElementById('header-sub').textContent = `Chatting as ${p.name}`;
    const tagsEl = document.getElementById('emp-tags');
    tagsEl.innerHTML = p.known_skills.slice(0, 3)
      .map(s => `<span class="tag">${escapeHtml(s)}</span>`).join('');

    loginScreen.classList.remove('active');
    chatScreen.classList.add('active');
    ensureIdleVideoPlaying();
    fetchWelcome(sessionId);

  } catch (err) {
    spin.classList.add('hidden');
    label.textContent = 'Start Conversation';
    btn.disabled = false;
    alert(`Could not connect to the HR Avatar backend.\n\nError: ${err.message}`);
  }
});

// ── Welcome ───────────────────────────────────────────────────────────────────
async function fetchWelcome(sid) {
  setProcessing(true);
  showThinking();
  try {
    const res  = await fetch(`${API}/session/welcome`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body:   JSON.stringify({ session_id: sid }),
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    removeThinking();
    if (data.video_url) {
      playVideoAndStreamText(data.video_url, data.reply);
    } else if (data.video_job_id) {
      waitForVideoThenPlay(data.video_job_id, data.reply);
    } else {
      addMessage('assistant', formatMessage(data.reply));
      setAvatar('idle');
    }
  } catch (_) {
    removeThinking();
    setAvatar('idle');
  } finally {
    setProcessing(false);
  }
}

// ── Text chat ─────────────────────────────────────────────────────────────────
sendBtn.addEventListener('click', submitText);
msgInput.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); submitText(); }
});
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
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body:   JSON.stringify({ session_id: sessionId, message: text }),
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    removeThinking();
    setAvatar('waiting');
    if (data.video_url) {
      playVideoAndStreamText(data.video_url, data.reply);
    } else if (data.video_job_id) {
      waitForVideoThenPlay(data.video_job_id, data.reply);
    } else {
      addMessage('assistant', formatMessage(data.reply));
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
    addMessage('user', escapeHtml(btn.dataset.q));
    callChat(btn.dataset.q);
  });
});

// ── Microphone VAD (Silero via @ricky0123/vad-web) ────────────────────────────
// Click mic → Silero VAD starts listening → auto-sends when speech ends
// Click mic again while active → cancel

micBtn.addEventListener('click', async () => {
  if (!sessionId || isProcessing) return;
  if (vadInstance) { cleanupVAD(); return; }
  await startVAD();
});

async function startVAD() {
  setMicListening();
  try {
    vadInstance = await vad.MicVAD.new({
      positiveSpeechThreshold: 0.8,
      negativeSpeechThreshold: 0.5,
      minSpeechFrames:         4,
      preSpeechPadFrames:      2,
      redemptionFrames:        8,
      onSpeechStart: () => {
        setMicRecording();
      },
      onSpeechEnd: (audio) => {
        // audio: Float32Array at 16 kHz — encode to WAV and send
        const wavBlob = float32ToWav(audio, 16000);
        cleanupVAD();
        sendAudio(wavBlob, 'audio/wav');
      },
      onVADMisfire: () => {
        // clip too short to be real speech — keep listening
        setMicListening();
      },
    });
    vadInstance.start();
  } catch (err) {
    console.error('[VAD] init failed:', err);
    alert('Microphone access denied or VAD failed to load.');
    setMicIdle();
  }
}

function cleanupVAD() {
  if (vadInstance) {
    try { vadInstance.destroy(); } catch (_) {}
    vadInstance = null;
  }
  setMicIdle();
}

function setMicListening() {
  micBtn.classList.remove('vad-off', 'vad-speaking');
  micBtn.classList.add('vad-listening');
  micBtn.title = 'Listening… click to cancel';
  recordingBar.classList.remove('hidden');
  recordingBar.classList.add('listening');
  const statusEl = document.getElementById('rec-status-text');
  if (statusEl) statusEl.textContent = 'Listening… speak now (stops automatically)';
}

function setMicRecording() {
  micBtn.classList.remove('vad-off', 'vad-listening');
  micBtn.classList.add('vad-speaking');
  micBtn.title = 'Recording… stops automatically when you finish speaking';
  recordingBar.classList.remove('hidden', 'listening');
  const statusEl = document.getElementById('rec-status-text');
  if (statusEl) statusEl.textContent = 'Hearing you…';
}

function setMicIdle() {
  micBtn.classList.remove('vad-speaking', 'vad-listening');
  micBtn.classList.add('vad-off');
  micBtn.title = 'Click to speak';
  recordingBar.classList.add('hidden');
  if (micLevelBar) micLevelBar.style.width = '0%';
}

// ── Audio send ────────────────────────────────────────────────────────────────
async function sendAudio(blob, mimeType) {
  setMicIdle();
  setProcessing(true);
  showThinking();

  try {
    const form = new FormData();
    form.append('session_id', sessionId);
    form.append('audio', blob, mimeType === 'audio/wav' ? 'recording.wav' : 'recording.webm');

    const res = await fetch(`${API}/chat/audio`, { method: 'POST', body: form });
    if (!res.ok) {
      const body = await res.text().catch(() => '');
      throw new Error(`HTTP ${res.status}${body ? ': ' + body : ''}`);
    }
    const data = await res.json();

    removeThinking();

    // Show what the user said in the chat
    const userLabel = data.transcription
      ? `🎤 <em>${escapeHtml(data.transcription)}</em>`
      : '🎤 <em>(voice message)</em>';
    addMessage('user', userLabel);

    setAvatar('waiting');
    if (data.video_url) {
      playVideoAndStreamText(data.video_url, data.reply);
    } else if (data.video_job_id) {
      waitForVideoThenPlay(data.video_job_id, data.reply);
    } else {
      addMessage('assistant', formatMessage(data.reply));
      setAvatar('idle');
    }
  } catch (err) {
    removeThinking();
    addMessage('error', `⚠️ Voice error: ${escapeHtml(err.message)}`);
    setAvatar('idle');
    console.error('[Mic] sendAudio error:', err);
  } finally {
    setProcessing(false);
  }
}

// ── End session ───────────────────────────────────────────────────────────────
document.getElementById('end-btn').addEventListener('click', async () => {
  if (!sessionId) return;
  if (!confirm('End this session and return to the login screen?')) return;
  cleanupVAD();
  await fetch(`${API}/session/${sessionId}`, { method: 'DELETE' }).catch(() => {});
  sessionId = null;
  chatScreen.classList.remove('active');
  loginScreen.classList.add('active');
  messagesEl.innerHTML = '';
  setAvatar('idle');
  setMicIdle();
  const btn   = document.getElementById('start-btn');
  const label = document.getElementById('start-btn-label');
  const spin  = document.getElementById('btn-spinner');
  btn.disabled = false;
  label.textContent = 'Start Conversation';
  spin.classList.add('hidden');
});
