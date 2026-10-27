// Tải MediaPipe Face Landmarker (tasks-vision) từ jsDelivr và chạy hoàn toàn trên máy.
export const MP_VERSION = '0.10.21';
export const MP_BASE = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MP_VERSION}`;
export const MODEL_URLS = [
  new URL('../models/face_landmarker.task', import.meta.url).href,
  'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task',
];

let visionMod = null, fileset = null, modelBuf = null;
const instances = {}, pending = {};

// MediaPipe in log thông tin ("INFO: Created TensorFlow Lite XNNPACK…") qua console.error – bỏ qua cho console sạch
{ const ce = console.error.bind(console); console.error = (...a) => { if (typeof a[0] === 'string' && /^(INFO|W\d{4}|I\d{4}):?\s/.test(a[0])) return; ce(...a); }; }

let modelP = null;
function fetchModel(onStatus) { return modelBuf ? Promise.resolve(modelBuf) : (modelP || (modelP = _fetchModel(onStatus).catch(e => { modelP = null; throw e; }))); }
async function _fetchModel(onStatus) {
  let lastErr;
  for (const url of MODEL_URLS) {
    try {
      const r = await fetch(url, { mode: 'cors' });
      if (!r.ok) throw new Error('HTTP ' + r.status);
      const b = new Uint8Array(await r.arrayBuffer());
      if (b.length < 1e6) throw new Error('model quá nhỏ');
      modelBuf = b; return b;
    } catch (e) { lastErr = e; onStatus && onStatus('Đang thử nguồn mô hình khác…'); }
  }
  throw lastErr || new Error('Không tải được mô hình');
}

async function create(mode, numFaces, delegate) {
  const { FaceLandmarker } = visionMod;
  return FaceLandmarker.createFromOptions(fileset, {
    baseOptions: { modelAssetBuffer: modelBuf, delegate },
    runningMode: mode,
    numFaces,
    minFaceDetectionConfidence: 0.4,
    minFacePresenceConfidence: 0.4,
    minTrackingConfidence: 0.4,
    outputFaceBlendshapes: false,
    outputFacialTransformationMatrixes: false,
  });
}

/** mode: 'IMAGE' | 'VIDEO' */
export function getLandmarker(mode = 'IMAGE', onStatus) {
  if (instances[mode]) return Promise.resolve(instances[mode]);
  return pending[mode] || (pending[mode] = _getLandmarker(mode, onStatus).finally(() => { delete pending[mode]; }));
}
async function _getLandmarker(mode, onStatus) {
  onStatus && onStatus('Đang tải bộ nhận diện khuôn mặt…');
  if (!visionMod) visionMod = await import(`${MP_BASE}/vision_bundle.mjs`);
  if (!fileset) fileset = await visionMod.FilesetResolver.forVisionTasks(`${MP_BASE}/wasm`);
  await fetchModel(onStatus);
  const n = mode === 'IMAGE' ? 6 : 1;
  let lm;
  try { lm = await create(mode, n, 'GPU'); }
  catch (e) { console.warn('GPU delegate lỗi, chuyển sang CPU', e); lm = await create(mode, n, 'CPU'); }
  instances[mode] = lm;
  // Báo service worker lưu sẵn wasm + mô hình để dùng offline
  try {
    const urls = [fileset.wasmLoaderPath, fileset.wasmBinaryPath, `${MP_BASE}/vision_bundle.mjs`, MODEL_URLS[0]];
    navigator.serviceWorker?.controller?.postMessage({ type: 'warm', urls });
    navigator.serviceWorker?.ready.then(r => r.active?.postMessage({ type: 'warm', urls }));
  } catch (_) {}
  return lm;
}

/** Trả về danh sách khuôn mặt: mỗi mặt là mảng điểm {x,y,z} theo pixel. */
export function toPixels(result, w, h) {
  return (result.faceLandmarks || []).map(face => face.map(p => ({ x: p.x * w, y: p.y * h, z: p.z * w })));
}

function boxOf(f) { let x0 = 1e9, y0 = 1e9, x1 = -1e9, y1 = -1e9; for (const p of f) { x0 = Math.min(x0, p.x); y0 = Math.min(y0, p.y); x1 = Math.max(x1, p.x); y1 = Math.max(y1, p.y); } return { x0, y0, x1, y1, cx: (x0 + x1) / 2, cy: (y0 + y1) / 2, s: Math.max(x1 - x0, y1 - y0) }; }

/**
 * Nhận diện trên toàn ảnh, rồi chạy lại trên vùng cắt độ phân giải cao quanh từng mặt
 * (mặt nhỏ trong ảnh lớn → landmark chính xác hơn nhiều).
 */
export async function detectImage(canvas, onStatus, { refine = true, maxRefine = 4 } = {}) {
  const lm = await getLandmarker('IMAGE', onStatus);
  const faces = toPixels(lm.detect(canvas), canvas.width, canvas.height);
  if (!refine || !faces.length) return faces;
  const R = 640, cc = document.createElement('canvas'); cc.width = cc.height = R;
  const ctx = cc.getContext('2d', { willReadFrequently: false });
  const order = faces.map((f, i) => [boxOf(f).s, i]).sort((a, b) => b[0] - a[0]).slice(0, maxRefine).map(v => v[1]);
  for (const i of order) {
    const b = boxOf(faces[i]); const side = b.s * 2.0, ox = b.cx - side / 2, oy = b.cy - side / 2;
    ctx.fillStyle = '#000'; ctx.fillRect(0, 0, R, R); ctx.imageSmoothingQuality = 'high';
    ctx.drawImage(canvas, ox, oy, side, side, 0, 0, R, R);
    let res; try { res = lm.detect(cc); } catch (_) { continue; }
    const cand = (res.faceLandmarks || []).map(f => f.map(p => ({ x: ox + p.x * side, y: oy + p.y * side, z: p.z * side })));
    let best = null, bd = 1e9;
    for (const f of cand) { const q = boxOf(f); const d = Math.hypot(q.cx - b.cx, q.cy - b.cy) / b.s + Math.abs(Math.log(q.s / b.s)); if (d < bd) { bd = d; best = f; } }
    if (best && bd < 0.35) faces[i] = best;
  }
  return faces;
}
