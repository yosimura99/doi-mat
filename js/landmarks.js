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

export async function detectImage(canvas, onStatus) {
  const lm = await getLandmarker('IMAGE', onStatus);
  const res = lm.detect(canvas);
  return toPixels(res, canvas.width, canvas.height);
}
