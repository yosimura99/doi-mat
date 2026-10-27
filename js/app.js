// Đổi Mặt – giao diện. Mọi xử lý diễn ra trên máy (MediaPipe + WebGL2), không tải ảnh lên.
import { getLandmarker, detectImage, toPixels } from './landmarks.js';
import { FaceSwapper, DEFAULTS } from './faceswap.js';
import { pickFace, bbox, IDX } from './geometry.js';

const $ = s => document.querySelector(s);
const MAX_SIDE = 4096, MAX_PIX = 16e6;
const isIOS = /iP(hone|ad|od)/.test(navigator.userAgent) || (navigator.platform === 'MacIntel' && navigator.maxTouchPoints > 1);

/* ---------- tiện ích ---------- */
const nextFrame = () => new Promise(r => requestAnimationFrame(() => setTimeout(r, 0)));
function setStatus(el, msg, err = false) { el.textContent = msg || ''; el.classList.toggle('err', !!err); }

async function fileToCanvas(file) {
  const url = URL.createObjectURL(file);
  try {
    const img = new Image(); img.decoding = 'async'; img.src = url; await img.decode();
    const w0 = img.naturalWidth, h0 = img.naturalHeight, k = Math.min(1, MAX_SIDE / Math.max(w0, h0), Math.sqrt(MAX_PIX / (w0 * h0)));
    const c = document.createElement('canvas'); c.width = Math.round(w0 * k); c.height = Math.round(h0 * k);
    const ctx = c.getContext('2d'); ctx.imageSmoothingQuality = 'high'; ctx.drawImage(img, 0, 0, c.width, c.height);
    return c;
  } finally { URL.revokeObjectURL(url); }
}

function cloneCanvas(src) { const c = document.createElement('canvas'); c.width = src.width; c.height = src.height; c.getContext('2d').drawImage(src, 0, 0); return c; }

function watermark(src, mirror = false) {
  const c = document.createElement('canvas'); c.width = src.width; c.height = src.height;
  const x = c.getContext('2d');
  if (mirror) { x.translate(c.width, 0); x.scale(-1, 1); }
  x.drawImage(src, 0, 0); x.setTransform(1, 0, 0, 1, 0, 0);
  const fs = Math.max(11, Math.round(Math.min(c.width, c.height) * 0.028)), pad = Math.round(fs * 0.8);
  x.font = `600 ${fs}px -apple-system, "Segoe UI", Roboto, Arial, sans-serif`; x.textAlign = 'right'; x.textBaseline = 'bottom';
  x.shadowColor = 'rgba(0,0,0,.45)'; x.shadowBlur = fs * 0.3; x.fillStyle = 'rgba(255,255,255,.55)';
  x.fillText('Face swap', c.width - pad, c.height - pad);
  return c;
}

async function saveCanvas(canvas, name = 'doi-mat.png') {
  const blob = await new Promise(r => canvas.toBlob(r, 'image/png'));
  const file = new File([blob], name, { type: 'image/png' });
  if (navigator.canShare && navigator.canShare({ files: [file] })) {
    try { await navigator.share({ files: [file], title: 'Đổi Mặt' }); return; }
    catch (e) { if (e.name === 'AbortError') return; }
  }
  const url = URL.createObjectURL(blob);
  if (isIOS) { // không chia sẻ được → hiện ảnh để nhấn giữ lưu
    $('#sheetImg').src = url; $('#sheetDl').href = url; $('#sheet').hidden = false; return;
  }
  const a = document.createElement('a'); a.href = url; a.download = name; document.body.appendChild(a); a.click(); a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 30000);
}
$('#sheetClose').onclick = () => { $('#sheet').hidden = true; };

/* ---------- tuỳ chọn nâng cao ---------- */
const optIds = ['identity', 'relight', 'color', 'feather', 'blend', 'keepMouth', 'keepEyes', 'occlusion'];
function readOpts() {
  const o = {};
  for (const k of optIds) { const el = $('#o-' + k); o[k] = el.type === 'checkbox' ? (k === 'occlusion' ? (el.checked ? 1 : 0) : el.checked) : el.tagName === 'SELECT' ? el.value : +el.value; }
  return o;
}
$('#o-reset').onclick = () => {
  for (const k of optIds) { const el = $('#o-' + k), v = k === 'occlusion' ? true : DEFAULTS[k]; if (el.type === 'checkbox') el.checked = !!v; else el.value = v; }
};

/* ---------- CHẾ ĐỘ ẢNH ---------- */
const slots = {
  A: { el: $('#slotA'), img: null, faces: [], sel: -1 },
  B: { el: $('#slotB'), img: null, faces: [], sel: -1 },
};
let swapper = null;
const getSwapper = () => swapper || (swapper = new FaceSwapper());

function drawThumb(s) {
  const body = s.el.querySelector('.slot-body'), cv = s.el.querySelector('.thumb'), badge = s.el.querySelector('.badge');
  if (!s.img) { cv.hidden = true; badge.hidden = true; s.el.querySelector('.empty').hidden = false; body.classList.remove('has'); return; }
  const k = Math.min(1, 720 / Math.max(s.img.width, s.img.height));
  cv.width = Math.round(s.img.width * k); cv.height = Math.round(s.img.height * k);
  const x = cv.getContext('2d'); x.drawImage(s.img, 0, 0, cv.width, cv.height);
  s.faces.forEach((f, i) => {
    const b = bbox(f, IDX.oval), on = i === s.sel;
    if (s.faces.length === 1 && on) return;
    x.lineWidth = Math.max(2, cv.width / 160); x.strokeStyle = on ? '#22d3ee' : 'rgba(255,255,255,.75)';
    x.setLineDash(on ? [] : [6, 5]);
    x.beginPath(); x.roundRect ? x.roundRect(b.x * k, b.y * k, b.w * k, b.h * k, 10) : x.rect(b.x * k, b.y * k, b.w * k, b.h * k); x.stroke();
  });
  cv.hidden = false; s.el.querySelector('.empty').hidden = true; body.classList.add('has');
  badge.hidden = false;
  badge.classList.toggle('bad', !s.faces.length);
  badge.textContent = s.busy ? 'Đang tìm khuôn mặt…' : !s.faces.length ? 'Không thấy khuôn mặt' : s.faces.length === 1 ? 'Đã nhận diện ✓' : `${s.faces.length} khuôn mặt – chạm để chọn`;
}

function updateGo() {
  $('#go').disabled = !(slots.A.sel >= 0 && slots.B.sel >= 0) || slots.A.busy || slots.B.busy;
  $('#liveUseA').hidden = !(slots.A.sel >= 0);
}

async function loadSlot(key, file) {
  const s = slots[key];
  try {
    s.busy = true; s.img = await fileToCanvas(file); s.faces = []; s.sel = -1; drawThumb(s); updateGo();
    setStatus($('#status'), 'Đang tìm khuôn mặt…');
    s.faces = await detectImage(s.img, m => setStatus($('#status'), m));
    s.sel = s.faces.length ? pickFace(s.faces, s.img.width, s.img.height) : -1;
    setStatus($('#status'), s.faces.length ? '' : `Không tìm thấy khuôn mặt trong ảnh ${key === 'A' ? 1 : 2}. Hãy chọn ảnh rõ mặt, đủ sáng.`, !s.faces.length);
  } catch (e) {
    console.error(e); setStatus($('#status'), 'Không mở được ảnh hoặc bộ nhận diện: ' + (e.message || e), true);
  } finally { s.busy = false; drawThumb(s); updateGo(); }
}

for (const key of ['A', 'B']) {
  const s = slots[key];
  s.el.querySelectorAll('input[type=file]').forEach(inp => inp.addEventListener('change', () => { if (inp.files[0]) loadSlot(key, inp.files[0]); inp.value = ''; }));
  const body = s.el.querySelector('.slot-body');
  body.addEventListener('click', ev => {
    if (!s.img) { s.el.querySelector('input[data-src=lib]').click(); return; }
    if (s.faces.length < 2) { s.el.querySelector('input[data-src=lib]').click(); return; }
    // chạm để chọn khuôn mặt (ảnh hiển thị kiểu contain)
    const cv = s.el.querySelector('.thumb'), r = cv.getBoundingClientRect();
    const k = Math.min(r.width / s.img.width, r.height / s.img.height), ox = (r.width - s.img.width * k) / 2, oy = (r.height - s.img.height * k) / 2;
    const px = (ev.clientX - r.left - ox) / k, py = (ev.clientY - r.top - oy) / k;
    let best = -1, bd = Infinity;
    s.faces.forEach((f, i) => { const b = bbox(f, IDX.oval); const d = Math.hypot(px - (b.x + b.w / 2), py - (b.y + b.h / 2)) / Math.max(b.w, b.h); if (d < bd) { bd = d; best = i; } });
    if (best >= 0 && bd < 1.2) { s.sel = best; drawThumb(s); updateGo(); }
  });
}
$('#flip').onclick = () => { const a = { ...slots.A }; for (const k of ['img', 'faces', 'sel']) { slots.A[k] = slots.B[k]; slots.B[k] = a[k]; } drawThumb(slots.A); drawThumb(slots.B); updateGo(); };

/* so sánh trước/sau */
let results = [], cur = 0;
function showResult(i) {
  cur = i; const r = results[i]; if (!r) return;
  const after = $('#after'), before = $('#before');
  after.width = r.after.width; after.height = r.after.height; after.getContext('2d').drawImage(r.after, 0, 0);
  before.width = r.before.width; before.height = r.before.height; before.getContext('2d').drawImage(r.before, 0, 0);
  document.querySelectorAll('.rtab').forEach(b => b.classList.toggle('on', +b.dataset.i === i));
}
function setSplit(v) {
  v = Math.max(0, Math.min(100, v)); $('#slider').value = v;
  $('#beforeWrap').style.clipPath = `inset(0 ${100 - v}% 0 0)`; $('#handle').style.left = v + '%';
}
$('#slider').addEventListener('input', e => setSplit(+e.target.value));
{
  const cmp = $('#compare'); let drag = false;
  const at = e => { const r = cmp.getBoundingClientRect(); setSplit((e.clientX - r.left) / r.width * 100); };
  cmp.addEventListener('pointerdown', e => { drag = true; at(e); });
  window.addEventListener('pointermove', e => drag && at(e));
  window.addEventListener('pointerup', () => { drag = false; });
}
document.querySelectorAll('.rtab').forEach(b => b.onclick = () => showResult(+b.dataset.i));

function swapOnce(src, dst, opts) {
  const fs = getSwapper();
  fs.setSource(src.img, src.faces[src.sel], src.img.width, src.img.height);
  const r = fs.render(dst.img, dst.img.width, dst.img.height, dst.faces[dst.sel], opts) || {};
  return { before: dst.img, after: cloneCanvas(fs.canvas), yaw: [r.yawSrc || 0, r.yawDst || 0] };
}

$('#go').onclick = async () => {
  const go = $('#go'); go.disabled = true;
  setStatus($('#status'), 'Đang đổi mặt…'); await nextFrame();
  try {
    const t0 = performance.now(), opts = readOpts();
    results = [swapOnce(slots.A, slots.B, opts)];
    if ($('#both').checked) { await nextFrame(); results.push(swapOnce(slots.B, slots.A, opts)); }
    $('#rtabs').hidden = results.length < 2;
    $('#result').hidden = false; showResult(0); setSplit(50);
    const [ya, yb] = results[0].yaw, tilted = Math.abs(ya - yb) > 15 || Math.max(Math.abs(ya), Math.abs(yb)) > 30;
    setStatus($('#status'), `Xong trong ${((performance.now() - t0) / 1000).toFixed(1)} giây ✓` +
      (tilted ? ' · Mẹo: ảnh chụp thẳng mặt và cùng góc nghiêng sẽ cho kết quả đẹp hơn.' : ''));
    $('#result').scrollIntoView({ behavior: 'smooth', block: 'start' });
  } catch (e) { console.error(e); setStatus($('#status'), 'Lỗi khi ghép mặt: ' + (e.message || e), true); }
  finally { updateGo(); }
};
$('#save').onclick = () => results[cur] && saveCanvas(watermark(results[cur].after), `doi-mat-${Date.now()}.png`);

/* ---------- CHẾ ĐỘ CAMERA ---------- */
const live = { swapper: null, stream: null, facing: 'user', raf: 0, lm: null, lastT: -1, fpsT: 0, fpsN: 0, srcReady: false };
const video = $('#video'), liveOut = $('#liveOut');

function setLiveSource(img, face) {
  if (!live.swapper) live.swapper = new FaceSwapper(liveOut);
  live.swapper.setSource(img, face, img.width, img.height);
  live.srcReady = true;
  const t = $('#liveThumb'), b = bbox(face, IDX.oval), m = Math.max(b.w, b.h) * 0.75, cx = b.x + b.w / 2, cy = b.y + b.h / 2;
  t.width = t.height = 128; t.getContext('2d').drawImage(img, cx - m, cy - m, 2 * m, 2 * m, 0, 0, 128, 128); t.hidden = false;
  $('#liveSrcInfo').textContent = 'Đã sẵn sàng. Bật camera để xem.';
}
$('#liveFile').addEventListener('change', async e => {
  const f = e.target.files[0]; e.target.value = ''; if (!f) return;
  try {
    setStatus($('#liveStatus'), 'Đang tìm khuôn mặt…');
    const img = await fileToCanvas(f), faces = await detectImage(img, m => setStatus($('#liveStatus'), m));
    if (!faces.length) { setStatus($('#liveStatus'), 'Không tìm thấy khuôn mặt trong ảnh này.', true); return; }
    setLiveSource(img, faces[pickFace(faces, img.width, img.height)]); setStatus($('#liveStatus'), '');
  } catch (err) { console.error(err); setStatus($('#liveStatus'), 'Lỗi: ' + (err.message || err), true); }
});
$('#liveUseA').onclick = () => { const s = slots.A; if (s.sel >= 0) setLiveSource(s.img, s.faces[s.sel]); };

async function startCam() {
  stopCam();
  setStatus($('#liveStatus'), 'Đang mở camera…');
  if (!navigator.mediaDevices?.getUserMedia) throw new Error('Trình duyệt không hỗ trợ camera (cần HTTPS)');
  live.stream = await navigator.mediaDevices.getUserMedia({ audio: false, video: { facingMode: live.facing, width: { ideal: 720 }, height: { ideal: 960 } } });
  video.srcObject = live.stream; await video.play();
  if (!live.swapper) live.swapper = new FaceSwapper(liveOut);
  live.lm = await getLandmarker('VIDEO', m => setStatus($('#liveStatus'), m));
  $('#stage').classList.toggle('mirror', live.facing === 'user');
  $('#stageEmpty').hidden = true; $('#fps').hidden = false; $('#shot').disabled = false; $('#camFlip').disabled = false;
  $('#camBtn').textContent = 'Tắt camera';
  setStatus($('#liveStatus'), live.srcReady ? '' : 'Chưa chọn mặt nguồn – đang hiện camera gốc.');
  live.fpsT = performance.now(); live.fpsN = 0; live.lastT = -1;
  loop();
}
function stopCam() {
  cancelAnimationFrame(live.raf); live.raf = 0;
  if (live.stream) live.stream.getTracks().forEach(t => t.stop());
  live.stream = null; video.srcObject = null;
  $('#stageEmpty').hidden = false; $('#fps').hidden = true; $('#shot').disabled = true; $('#camFlip').disabled = true;
  $('#camBtn').textContent = 'Bật camera';
}
function loop() {
  live.raf = requestAnimationFrame(loop);
  if (video.readyState < 2 || video.currentTime === live.lastT) return;
  live.lastT = video.currentTime;
  const vw = video.videoWidth, vh = video.videoHeight; if (!vw) return;
  let face = null;
  if (live.srcReady) {
    try { face = toPixels(live.lm.detectForVideo(video, performance.now()), vw, vh)[0] || null; } catch (e) { console.warn(e); }
  }
  live.swapper.render(video, vw, vh, face, { proc: 448, statsEvery: 8, blend: 'feather', feather: 1.3, grain: true, identity: +$('#o-identity').value });
  live.fpsN++;
  const now = performance.now();
  if (now - live.fpsT > 600) { $('#fps').textContent = `${Math.round(live.fpsN * 1000 / (now - live.fpsT))} FPS`; live.fpsT = now; live.fpsN = 0; }
}
$('#camBtn').onclick = async () => {
  if (live.stream) { stopCam(); setStatus($('#liveStatus'), ''); return; }
  try { await startCam(); } catch (e) { console.error(e); stopCam(); setStatus($('#liveStatus'), e.name === 'NotAllowedError' ? 'Bạn chưa cho phép dùng camera. Vào Cài đặt › Safari › Camera để bật.' : 'Không mở được camera: ' + (e.message || e), true); }
};
$('#camFlip').onclick = async () => { live.facing = live.facing === 'user' ? 'environment' : 'user'; try { await startCam(); } catch (e) { setStatus($('#liveStatus'), 'Không đổi được camera: ' + e.message, true); } };
$('#shot').onclick = () => saveCanvas(watermark(liveOut, live.facing === 'user'), `doi-mat-${Date.now()}.png`);

/* ---------- tab ---------- */
document.querySelectorAll('.tab').forEach(t => t.onclick = () => {
  const name = t.dataset.tab;
  document.querySelectorAll('.tab').forEach(x => { x.classList.toggle('on', x === t); x.setAttribute('aria-selected', x === t); });
  $('#tab-photo').hidden = name !== 'photo'; $('#tab-live').hidden = name !== 'live';
  if (name !== 'live') stopCam();
  else if (!live.srcReady && slots.A.sel >= 0) setLiveSource(slots.A.img, slots.A.faces[slots.A.sel]);
});
document.addEventListener('visibilitychange', () => { if (document.hidden && live.stream) stopCam(); });

/* ---------- PWA ---------- */
if ('serviceWorker' in navigator) {
  navigator.serviceWorker.register('sw.js').catch(e => console.warn('SW', e));
  navigator.serviceWorker.addEventListener('message', e => { if (e.data?.type === 'warmed') $('#offline').textContent = '• Đã sẵn sàng dùng offline.'; });
}
// tải trước bộ nhận diện khi rảnh để lần đầu nhanh hơn
(window.requestIdleCallback || (f => setTimeout(f, 1200)))(() => getLandmarker('IMAGE').catch(e => console.warn('preload', e)));
updateGo();
window.__doiMat = { slots, loadSlot, results: () => results };
