// Service worker Đổi Mặt: network-first (luôn lấy bản mới khi có mạng), rơi về cache khi offline.
// Ngoài giao diện, còn lưu wasm MediaPipe + mô hình sau lần dùng đầu để chạy offline.
const VERSION = 'doi-mat-v2';
const SHELL = ['./', 'index.html', 'css/app.css', 'js/app.js', 'js/landmarks.js', 'js/faceswap.js', 'js/geometry.js', 'js/canonical.js',
  'manifest.webmanifest', 'icons/icon.svg', 'icons/icon-192.png', 'icons/icon-512.png', 'icons/apple-touch-icon.png'];
const CACHEABLE = u => u.origin === self.location.origin || u.hostname === 'cdn.jsdelivr.net' || u.hostname === 'storage.googleapis.com';

self.addEventListener('install', e => {
  e.waitUntil(caches.open(VERSION).then(c => c.addAll(SHELL)).then(() => self.skipWaiting()));
});
self.addEventListener('activate', e => {
  e.waitUntil(caches.keys().then(ks => Promise.all(ks.filter(k => k !== VERSION).map(k => caches.delete(k)))).then(() => self.clients.claim()));
});

async function networkFirst(req) {
  const cache = await caches.open(VERSION);
  try {
    const res = await fetch(req);
    if (res && (res.ok || res.type === 'opaque')) cache.put(req, res.clone()).catch(() => {});
    return res;
  } catch (err) {
    const hit = await cache.match(req, { ignoreSearch: req.mode === 'navigate' });
    if (hit) return hit;
    if (req.mode === 'navigate') { const idx = await cache.match('index.html'); if (idx) return idx; }
    throw err;
  }
}

self.addEventListener('fetch', e => {
  const req = e.request;
  if (req.method !== 'GET') return;
  const u = new URL(req.url);
  if (!CACHEABLE(u)) return;
  e.respondWith(networkFirst(req));
});

// Trang báo danh sách file MediaPipe/mô hình cần lưu sẵn
self.addEventListener('message', e => {
  if (e.data?.type !== 'warm') return;
  const urls = (e.data.urls || []).filter(Boolean);
  e.waitUntil((async () => {
    const cache = await caches.open(VERSION);
    let ok = 0;
    for (const url of urls) {
      try {
        if (await cache.match(url)) { ok++; continue; }
        const r = await fetch(url, { mode: 'cors' });
        if (r.ok) { await cache.put(url, r); ok++; }
      } catch (_) {}
    }
    if (ok) (await self.clients.matchAll()).forEach(c => c.postMessage({ type: 'warmed', ok }));
  })());
});
