// Hình học khuôn mặt: Delaunay, căn chỉnh 3D (Horn), độ hiển thị và trọng số đối xứng.
import { CANON, MIRROR } from './canonical.js';

export const IDX = {
  oval: [10,338,297,332,284,251,389,356,454,323,361,288,397,365,379,378,400,377,152,148,176,149,150,136,172,58,132,93,234,127,162,21,54,103,67,109],
  leftEye: [263,249,390,373,374,380,381,382,362,398,384,385,386,387,388,466],
  rightEye: [33,7,163,144,145,153,154,155,133,173,157,158,159,160,161,246],
  leftBrow: [276,283,282,295,285,300,293,334,296,336],
  rightBrow: [46,53,52,65,55,70,63,105,66,107],
  lipsOuter: [61,146,91,181,84,17,314,405,321,375,291,409,270,269,267,0,37,39,40,185],
  lipsInner: [78,95,88,178,87,14,317,402,318,324,308,415,310,311,312,13,82,81,80,191],
};
export const N = 468;

/** Bowyer–Watson Delaunay. pts: Float32Array [x0,y0,x1,y1,...] → Uint16Array tam giác */
export function delaunay(pts) {
  const n = pts.length / 2;
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  for (let i = 0; i < n; i++) { const x = pts[2*i], y = pts[2*i+1]; if (x < minX) minX = x; if (y < minY) minY = y; if (x > maxX) maxX = x; if (y > maxY) maxY = y; }
  const d = Math.max(maxX - minX, maxY - minY) * 20, cx = (minX + maxX) / 2, cy = (minY + maxY) / 2;
  const P = Array.from({ length: n }, (_, i) => [pts[2*i], pts[2*i+1]]);
  P.push([cx - d, cy - d], [cx + d, cy - d], [cx, cy + d]);
  const circ = (a, b, c) => {
    const [ax, ay] = P[a], [bx, by] = P[b], [cx2, cy2] = P[c];
    const D = 2 * (ax * (by - cy2) + bx * (cy2 - ay) + cx2 * (ay - by));
    const ux = ((ax*ax+ay*ay)*(by-cy2) + (bx*bx+by*by)*(cy2-ay) + (cx2*cx2+cy2*cy2)*(ay-by)) / D;
    const uy = ((ax*ax+ay*ay)*(cx2-bx) + (bx*bx+by*by)*(ax-cx2) + (cx2*cx2+cy2*cy2)*(bx-ax)) / D;
    return { a, b, c, x: ux, y: uy, r2: (ax-ux)**2 + (ay-uy)**2 };
  };
  let tris = [circ(n, n+1, n+2)];
  for (let i = 0; i < n; i++) {
    const [px, py] = P[i];
    const bad = [], keep = [];
    for (const t of tris) ((px-t.x)**2 + (py-t.y)**2 < t.r2 ? bad : keep).push(t);
    const edges = new Map();
    for (const t of bad) for (const [u, v] of [[t.a,t.b],[t.b,t.c],[t.c,t.a]]) {
      const k = u < v ? u + ',' + v : v + ',' + u;
      edges.has(k) ? edges.delete(k) : edges.set(k, [u, v]);
    }
    for (const [u, v] of edges.values()) keep.push(circ(u, v, i));
    tris = keep;
  }
  const out = [];
  for (const t of tris) if (t.a < n && t.b < n && t.c < n) {
    const [ax, ay] = P[t.a], [bx, by] = P[t.b], [qx, qy] = P[t.c];
    const area = (bx-ax)*(qy-ay) - (by-ay)*(qx-ax);
    if (Math.abs(area) < 1e-7) continue;
    // chuẩn hoá chiều quay (cùng dấu cho mọi tam giác)
    area > 0 ? out.push(t.a, t.b, t.c) : out.push(t.a, t.c, t.b);
  }
  return new Uint16Array(out);
}

let _tris = null, _inner = null, _upper = null;
export function triangles() { return _tris || (_tris = delaunay(CANON)); }

/** Trọng số "phần trong" của mặt (0 ở viền mặt, 1 ở giữa) – để giữ đường viền hàm của ảnh đích */
export function innerWeights() {
  if (_inner) return _inner;
  const w = new Float32Array(N); let R = 0;
  const d = new Float32Array(N);
  for (let i = 0; i < N; i++) {
    let m = Infinity;
    for (const j of IDX.oval) m = Math.min(m, Math.hypot(CANON[2*i]-CANON[2*j], CANON[2*i+1]-CANON[2*j+1]));
    d[i] = m; R = Math.max(R, m);
  }
  for (let i = 0; i < N; i++) w[i] = smooth(0.06, 0.35, d[i] / R);
  return (_inner = w);
}

export function smooth(a, b, x) { const t = Math.min(1, Math.max(0, (x - a) / (b - a))); return t * t * (3 - 2 * t); }

/** Horn (quaternion) – tìm s,R,t sao cho B ≈ s·R·A + t (3D) */
export function horn(A, B, weights) {
  let W = 0; const ca = [0,0,0], cb = [0,0,0];
  for (let i = 0; i < A.length; i++) { const w = weights ? weights[i] : 1; W += w; for (let k = 0; k < 3; k++) { ca[k] += w*A[i][k]; cb[k] += w*B[i][k]; } }
  for (let k = 0; k < 3; k++) { ca[k] /= W; cb[k] /= W; }
  const S = [[0,0,0],[0,0,0],[0,0,0]]; let na = 0, nb = 0;
  for (let i = 0; i < A.length; i++) {
    const w = weights ? weights[i] : 1;
    const a = [A[i][0]-ca[0], A[i][1]-ca[1], A[i][2]-ca[2]], b = [B[i][0]-cb[0], B[i][1]-cb[1], B[i][2]-cb[2]];
    for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++) S[r][c] += w * a[r] * b[c];
    na += w * (a[0]*a[0]+a[1]*a[1]+a[2]*a[2]); nb += w * (b[0]*b[0]+b[1]*b[1]+b[2]*b[2]);
  }
  const [[xx,xy,xz],[yx,yy,yz],[zx,zy,zz]] = S;
  const Nm = [
    [xx+yy+zz, yz-zy, zx-xz, xy-yx],
    [yz-zy, xx-yy-zz, xy+yx, zx+xz],
    [zx-xz, xy+yx, -xx+yy-zz, yz+zy],
    [xy-yx, zx+xz, yz+zy, -xx-yy+zz]];
  const q = maxEigvec4(Nm);
  const [q0, q1, q2, q3] = q;
  const R = [
    [q0*q0+q1*q1-q2*q2-q3*q3, 2*(q1*q2-q0*q3), 2*(q1*q3+q0*q2)],
    [2*(q1*q2+q0*q3), q0*q0-q1*q1+q2*q2-q3*q3, 2*(q2*q3-q0*q1)],
    [2*(q1*q3-q0*q2), 2*(q2*q3+q0*q1), q0*q0-q1*q1-q2*q2+q3*q3]];
  const s = Math.sqrt(nb / Math.max(na, 1e-9));
  const t = [0,1,2].map(r => cb[r] - s * (R[r][0]*ca[0] + R[r][1]*ca[1] + R[r][2]*ca[2]));
  return { s, R, t, apply: p => [0,1,2].map(r => s * (R[r][0]*p[0] + R[r][1]*p[1] + R[r][2]*p[2]) + t[r]) };
}

function maxEigvec4(M) { // Jacobi
  const a = M.map(r => r.slice()); const v = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]];
  for (let sweep = 0; sweep < 30; sweep++) {
    let off = 0; for (let p = 0; p < 4; p++) for (let q = p+1; q < 4; q++) off += a[p][q]*a[p][q];
    if (off < 1e-18) break;
    for (let p = 0; p < 4; p++) for (let q = p+1; q < 4; q++) {
      if (Math.abs(a[p][q]) < 1e-30) continue;
      const th = (a[q][q] - a[p][p]) / (2 * a[p][q]);
      const t = Math.sign(th || 1) / (Math.abs(th) + Math.sqrt(th*th + 1));
      const c = 1 / Math.sqrt(t*t + 1), s = t * c;
      for (let k = 0; k < 4; k++) { const akp = a[k][p], akq = a[k][q]; a[k][p] = c*akp - s*akq; a[k][q] = s*akp + c*akq; }
      for (let k = 0; k < 4; k++) { const apk = a[p][k], aqk = a[q][k]; a[p][k] = c*apk - s*aqk; a[q][k] = s*apk + c*aqk; }
      for (let k = 0; k < 4; k++) { const vkp = v[k][p], vkq = v[k][q]; v[k][p] = c*vkp - s*vkq; v[k][q] = s*vkp + c*vkq; }
    }
  }
  let best = 0; for (let i = 1; i < 4; i++) if (a[i][i] > a[best][best]) best = i;
  const q = [v[0][best], v[1][best], v[2][best], v[3][best]];
  const n = Math.hypot(...q); return q.map(x => x / n);
}

/** Độ "nhìn thẳng" của từng đỉnh (cos góc giữa pháp tuyến và camera) dựa trên lưới 3D */
export function vertexVisibility(lm) {
  const T = triangles(), vis = new Float32Array(N), cnt = new Float32Array(N);
  for (let t = 0; t < T.length; t += 3) {
    const a = lm[T[t]], b = lm[T[t+1]], c = lm[T[t+2]];
    const ux = b.x-a.x, uy = b.y-a.y, uz = b.z-a.z, vx = c.x-a.x, vy = c.y-a.y, vz = c.z-a.z;
    const nx = uy*vz-uz*vy, ny = uz*vx-ux*vz, nz = ux*vy-uy*vx;
    const v = nz / (Math.hypot(nx, ny, nz) || 1); // canonical winding cho nz>0 khi nhìn thẳng
    for (const i of [T[t], T[t+1], T[t+2]]) { vis[i] += v; cnt[i]++; }
  }
  for (let i = 0; i < N; i++) vis[i] = cnt[i] ? vis[i] / cnt[i] : 1;
  return vis;
}

/** Trọng số lấy mẫu từ nửa mặt đối xứng khi nửa mặt nguồn bị quay khuất */
export function mirrorWeights(vis) {
  const w = new Float32Array(N);
  for (let i = 0; i < N; i++) {
    const gain = vis[MIRROR[i]] - vis[i];
    w[i] = smooth(0.12, 0.4, gain) * (1 - smooth(0.6, 0.8, vis[i]));
  }
  // làm mượt theo lưới
  const T = triangles(); const acc = new Float32Array(N), c = new Float32Array(N);
  for (let t = 0; t < T.length; t += 3) { const m = (w[T[t]] + w[T[t+1]] + w[T[t+2]]) / 3; for (let k = 0; k < 3; k++) { acc[T[t+k]] += m; c[T[t+k]]++; } }
  for (let i = 0; i < N; i++) w[i] = 0.5 * w[i] + 0.5 * (c[i] ? acc[i] / c[i] : 0);
  return w;
}

export { MIRROR };

export function bbox(lm, idx) {
  let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
  for (const i of idx || lm.keys()) { const p = lm[i]; if (p.x < x0) x0 = p.x; if (p.y < y0) y0 = p.y; if (p.x > x1) x1 = p.x; if (p.y > y1) y1 = p.y; }
  return { x: x0, y: y0, w: x1 - x0, h: y1 - y0 };
}

export function openness(lm) {
  const d = (a, b) => Math.hypot(lm[a].x - lm[b].x, lm[a].y - lm[b].y);
  return {
    mouth: d(13, 14) / (d(61, 291) || 1),
    eyeR: d(159, 145) / (d(33, 133) || 1),
    eyeL: d(386, 374) / (d(362, 263) || 1),
  };
}

/** Chọn mặt tự động: lớn nhất + gần giữa ảnh */
export function pickFace(faces, w, h) {
  let best = 0, bs = -Infinity;
  faces.forEach((f, i) => {
    const b = bbox(f, IDX.oval); const area = b.w * b.h;
    const dc = Math.hypot((b.x + b.w/2) / w - 0.5, (b.y + b.h/2) / h - 0.5);
    const s = area * (1 - 0.8 * dc);
    if (s > bs) { bs = s; best = i; }
  });
  return best;
}
