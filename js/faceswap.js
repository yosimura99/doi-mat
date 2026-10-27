// Bộ ghép mặt WebGL2: warp tam giác Delaunay → chuyển ánh sáng tần số thấp → khớp màu LAB (vùng da)
// → khớp độ nét/hạt → trộn đa băng tần (Laplacian pyramid) với mặt nạ bao lồi đã co + làm mờ viền.
import { IDX, N, triangles, horn, vertexVisibility, mirrorWeights, innerWeights, MIRROR, openness, bbox, smooth, yawOf } from './geometry.js';

const VS_QUAD = `#version 300 es
in vec2 a_q; uniform float u_flip; out vec2 v_uv;
void main(){ v_uv=a_q; gl_Position=vec4(a_q.x*2.-1., u_flip>0.5 ? 1.-2.*a_q.y : 2.*a_q.y-1., 0., 1.); }`;

const VS_WARP = `#version 300 es
in vec3 a_pos; in vec2 a_uv; in vec2 a_uvm; in vec2 a_w;
out vec2 v_uv; out vec2 v_uvm; out vec2 v_w;
void main(){ v_uv=a_uv; v_uvm=a_uvm; v_w=a_w; gl_Position=vec4(a_pos.x*2.-1., a_pos.y*2.-1., a_pos.z, 1.); }`;

const H = `#version 300 es
precision highp float; in vec2 v_uv; out vec4 o;
`;
const LAB = `
vec3 toLin(vec3 c){ return mix(c/12.92, pow((c+0.055)/1.055, vec3(2.4)), step(0.04045,c)); }
vec3 toSrgb(vec3 c){ c=clamp(c,0.,1.); return mix(c*12.92, 1.055*pow(c,vec3(1./2.4))-0.055, step(0.0031308,c)); }
float fL(float t){ return t>0.008856 ? pow(t,1./3.) : 7.787*t+16./116.; }
float fLi(float t){ return t>0.206897 ? t*t*t : (t-16./116.)/7.787; }
vec3 rgb2lab(vec3 c){ vec3 l=toLin(c);
  float X=dot(l,vec3(0.4124,0.3576,0.1805))/0.95047, Y=dot(l,vec3(0.2126,0.7152,0.0722)), Z=dot(l,vec3(0.0193,0.1192,0.9505))/1.08883;
  float fx=fL(X), fy=fL(Y), fz=fL(Z); return vec3(116.*fy-16., 500.*(fx-fy), 200.*(fy-fz)); }
vec3 lab2rgb(vec3 c){ float fy=(c.x+16.)/116., fx=fy+c.y/500., fz=fy-c.z/200.;
  vec3 xyz=vec3(fLi(fx)*0.95047, fLi(fy), fLi(fz)*1.08883);
  vec3 l=vec3(dot(xyz,vec3(3.2406,-1.5372,-0.4986)), dot(xyz,vec3(-0.9689,1.8758,0.0415)), dot(xyz,vec3(0.0557,-0.2040,1.0570)));
  return toSrgb(l); }
float luma(vec3 c){ return dot(c, vec3(0.299,0.587,0.114)); }
`;

const FS = {
  warp: `#version 300 es
precision highp float; in vec2 v_uv; in vec2 v_uvm; in vec2 v_w; uniform sampler2D u_src; out vec4 o;
// alpha = độ tin cậy của mặt nguồn tại đỉnh (tam giác bị quay khuất ở cả hai nửa → để da ảnh đích)
void main(){ vec3 a=texture(u_src,v_uv).rgb, b=texture(u_src,v_uvm).rgb; o=vec4(mix(a,b,clamp(v_w.x,0.,1.)),clamp(v_w.y,0.,1.)); }`,
  crop: H + `uniform sampler2D u_t; uniform vec4 u_crop; void main(){ o=vec4(texture(u_t, u_crop.xy + v_uv*u_crop.zw).rgb,1.); }`,
  copy: H + `uniform sampler2D u_t; void main(){ o=texture(u_t, v_uv); }`,
  blur: H + `uniform sampler2D u_t; uniform vec2 u_dir; uniform float u_sigma;
void main(){ vec4 s=vec4(0.); float ws=0.; int r=int(min(ceil(u_sigma*2.5),40.));
  for(int i=-40;i<=40;i++){ if(i<-r||i>r) continue; float fi=float(i); float w=exp(-fi*fi/(2.*u_sigma*u_sigma)); s+=texture(u_t,v_uv+u_dir*fi)*w; ws+=w; }
  o=s/ws; }`,
  // Che khuất (tóc mái, tay…) ở vùng trán/thái dương: pixel khác xa màu da → giữ ảnh đích
  occ: H + LAB + `uniform sampler2D u_w; uniform sampler2D u_d; uniform sampler2D u_z; uniform vec3 u_sm; uniform vec3 u_ss; uniform vec3 u_dm; uniform vec3 u_ds; uniform vec2 u_ec; uniform vec2 u_up; uniform sampler2D u_sl; uniform sampler2D u_wm; uniform sampler2D u_slm;
void main(){ vec4 w=texture(u_w,v_uv); vec3 d=texture(u_d,v_uv).rgb; vec4 sl=texture(u_sl,v_uv); vec4 wm=texture(u_wm,v_uv); vec4 slm=texture(u_slm,v_uv);
  vec3 qs=(rgb2lab(w.rgb)-u_sm)/u_ss, qd=(rgb2lab(d)-u_dm)/u_ds; vec3 k=vec3(0.35,1.,1.);
  // bóng đổ trên da thì đậm màu hơn (a,b tăng), tóc thì nhạt màu → lệch dương của a,b bị phạt nhẹ
  vec3 qs2=vec3(qs.x, qs.y>0.?qs.y*0.4:qs.y, qs.z>0.?qs.z*0.4:qs.z);
  float d2s=dot(qs2*qs2,k), d2d=dot(qd*qd,k);
  float t=dot(v_uv-u_ec,u_up); float zone=max(smoothstep(0.05,0.28,t), texture(u_z,v_uv).r);
  float strand=0.; if(sl.a>0.003){ vec3 lr=rgb2lab(sl.rgb/sl.a), lw=rgb2lab(w.rgb); strand=smoothstep(5.,11.,lr.x-lw.x)*smoothstep(0.95,0.75,length(lw.yz)/max(length(lr.yz),1.)); }
  float zs=max(zone, texture(u_z,v_uv).r);
  vec3 qm=(rgb2lab(wm.rgb)-u_sm)/u_ss; float d2m=dot(qm*qm,k);
  float strandM=0.; if(slm.a>0.003){ strandM=smoothstep(7.,15.,rgb2lab(slm.rgb/slm.a).x-rgb2lab(wm.rgb).x); }
  float mok=(1.-smoothstep(9.,20.,d2m))*(1.-strandM)*smoothstep(0.12,0.45,wm.a);
  o=vec4(max(smoothstep(6.,14.,d2s)*zone, strand*zs)*w.a, smoothstep(9.,24.,d2d)*zone, mok, 1.); }`,
  // da cục bộ của nguồn (chuẩn hoá theo trọng số "giống da") → tham chiếu phát hiện sợi tóc và để vá
  sprep: H + LAB + `uniform sampler2D u_w; uniform sampler2D u_m; uniform vec3 u_sm; uniform vec3 u_ss;
void main(){ vec4 w=texture(u_w,v_uv); vec4 m=texture(u_m,v_uv); vec3 q=(rgb2lab(w.rgb)-u_sm)/u_ss;
  float d2=dot(q*q,vec3(0.35,1.,1.)); float dark=max(0.,-q.x);
  float ws=m.r*w.a*(1.-smoothstep(3.,8.,d2))*(1.-smoothstep(1.2,2.4,dark)); o=vec4(w.rgb*ws, ws); }`,
  // vá chỗ tóc/vật che trên mặt nguồn bằng da cục bộ của chính mặt nguồn
  // ưu tiên nửa mặt đối xứng (đã cân độ sáng cục bộ); sợi tóc mảnh → da cục bộ; còn lại → để ảnh đích (mcomb)
  inpaint: H + `uniform sampler2D u_w; uniform sampler2D u_s; uniform sampler2D u_o; uniform sampler2D u_wm; uniform sampler2D u_sm2; uniform float u_hasM;
void main(){ vec4 w=texture(u_w,v_uv); vec4 s=texture(u_s,v_uv); vec3 oc=texture(u_o,v_uv).rgb; float k=oc.r;
  vec3 col=w.rgb;
  float ks=k*smoothstep(0.01,0.06,s.a); vec3 ref = s.a>1e-3 ? s.rgb/s.a : w.rgb;
  if(u_hasM>0.5){ vec4 m=texture(u_wm,v_uv); vec4 sm=texture(u_sm2,v_uv);
    vec3 refm = sm.a>1e-3 ? sm.rgb/sm.a : m.rgb;
    vec3 g = s.a>0.02 ? clamp(ref/max(refm,vec3(0.02)), vec3(0.6), vec3(1.6)) : vec3(1.);
    float km=k*oc.b; col=mix(col, ref, ks*(1.-oc.b)); col=mix(col, clamp(m.rgb*g,0.,1.), km); }
  else col=mix(col, ref, ks);
  o=vec4(col, w.a); }`,
  mcomb: H + `uniform sampler2D u_m; uniform sampler2D u_o; uniform sampler2D u_o2; uniform float u_amt; uniform float u_big;
void main(){ vec4 m=texture(u_m,v_uv); vec2 c=texture(u_o,v_uv).rg; vec3 cc=texture(u_o,v_uv).rgb; vec3 c2=texture(u_o2,v_uv).rgb;
  // nới rộng vùng tóc che của nguồn (bắt cả sợi tóc mảnh sát mảng tóc)
  cc.x=max(cc.x, smoothstep(0.1,0.4,c2.x)*(1.-c2.z)); float k=1.-clamp(max(cc.y, cc.x*(1.-cc.z)*u_big)*1.6,0.,1.)*u_amt; o=vec4(m.r*k, m.g*(1.-c.x), m.b*(1.-c.y), 1.); }`,
  // Ước lượng ánh sáng (chuẩn hoá theo vùng da, loại điểm khác màu da): (Yđ·wđ, wđ, Yn·wn, wn)
  illumPrep: H + LAB + `uniform sampler2D u_d; uniform sampler2D u_w; uniform sampler2D u_m; uniform vec3 u_sm; uniform vec3 u_ss; uniform vec3 u_dm; uniform vec3 u_ds;
void main(){ vec4 w=texture(u_w,v_uv); vec3 d=texture(u_d,v_uv).rgb; vec4 m=texture(u_m,v_uv);
  vec3 qs=(rgb2lab(w.rgb)-u_sm)/u_ss, qd=(rgb2lab(d)-u_dm)/u_ds;
  // nguồn: KHÔNG loại vùng tối theo L (đó chính là bóng cần khử), chỉ loại theo sắc độ (tóc, nền)
  float ws=m.g*w.a*(1.-smoothstep(6.,18.,dot(qs*qs,vec3(0.04,1.,1.)))), wd=m.b*(1.-smoothstep(6.,18.,dot(qd*qd,vec3(0.15,1.,1.))));
  o=vec4(rgb2lab(d).x*0.01*wd, wd, rgb2lab(w.rgb).x*0.01*ws, ws); }`,
  // Thay bóng/ánh sáng tần số thấp của nguồn bằng của ảnh đích
  // trong không gian LAB: bỏ bóng tần số thấp của nguồn (L và cả độ "ấm" đi kèm theo độ dốc a,b~L), thêm bóng của ảnh đích
  relight: H + LAB + `uniform sampler2D u_w; uniform sampler2D u_i; uniform float u_strength; uniform vec3 u_sm; uniform vec3 u_dm; uniform vec2 u_ks; uniform vec2 u_kt; uniform float u_lim;
void main(){ vec4 w=texture(u_w,v_uv); vec4 i=texture(u_i,v_uv);
  float ok=smoothstep(0.02,0.2,min(i.y,i.w))*u_strength;
  float Lt=100.*(i.x+1e-5)/(i.y+1e-5), Ls=100.*(i.z+1e-5)/(i.w+1e-5);
  float ds=clamp(Ls-u_sm.x,-u_lim,u_lim)*ok, dt=clamp(Lt-u_dm.x,-u_lim,u_lim)*ok;
  vec3 l=rgb2lab(w.rgb);
  // bóng tối thì nhân mạnh hơn (tỉ lệ theo L), vùng sáng cộng gần như tuyến tính
  float g=clamp(l.x/max(Ls,5.),0.5,1.6);
  float L0=l.x; l.x=max(0.,l.x+(dt-ds)*g);
  l.yz=l.yz*pow(clamp(l.x/max(L0,1.),0.3,1.15),0.6)+clamp((u_kt*dt-u_ks*ds)*g, vec2(-10.), vec2(10.));
  o=vec4(clamp(lab2rgb(l),0.,1.), w.a); }`,
  statsPack: H + `uniform sampler2D u_t; uniform sampler2D u_m; uniform float u_useA; uniform float u_ch;
void main(){ vec4 c=texture(u_t,v_uv); vec4 mm=texture(u_m,v_uv); float m=u_ch>1.5?mm.b:mm.g; o=vec4(c.rgb, m*mix(1.,c.a,u_useA)); }`,
  labXfer: H + LAB + `uniform sampler2D u_w; uniform vec3 u_ms; uniform vec3 u_ss; uniform vec3 u_mt; uniform vec3 u_st; uniform float u_amt; uniform vec2 u_ks; uniform vec2 u_kt;
void main(){ vec4 w=texture(u_w,v_uv); vec3 l=rgb2lab(w.rgb); vec3 r=clamp(u_st/max(u_ss,vec3(1e-3)), vec3(0.7,0.6,0.6), vec3(1.4,1.3,1.3));
  float tl=(l.x-u_ms.x)*r.x+u_mt.x;
  // a,b: bỏ phần phụ thuộc độ sáng của nguồn, khớp phần dư, rồi cộng lại phụ thuộc độ sáng của ảnh đích
  vec2 res=(l.yz-u_ms.yz-u_ks*(l.x-u_ms.x))*r.yz;
  vec3 t=vec3(tl, u_mt.yz+res+u_kt*(tl-u_mt.x));
  // không để phép chuyển màu da đẩy điểm ít màu (tròng mắt, lòng trắng) sang xanh lạnh
  t.yz=max(t.yz, min(l.yz, vec2(1.5))); o=vec4(lab2rgb(mix(l,t,u_amt)), w.a); }`,
  measure: H + LAB + `uniform sampler2D u_a; uniform sampler2D u_d; uniform sampler2D u_m;
void main(){ vec4 a=texture(u_a,v_uv); o=vec4(luma(a.rgb), luma(texture(u_d,v_uv).rgb), texture(u_m,v_uv).g*a.a, 1.); }`,
  // Khớp độ nét + hạt nhiễu, rồi đặt lên ảnh đích theo vùng phủ của warp
  finalA: H + LAB + `uniform sampler2D u_c; uniform sampler2D u_d; uniform vec2 u_px; uniform float u_sharp; uniform float u_noise; uniform float u_seed;
uniform sampler2D u_o; uniform sampler2D u_o2; uniform sampler2D u_mz; uniform float u_fill;
float hash(vec2 p){ p=fract(p*vec2(443.897,441.423)+u_seed); p+=dot(p,p.yx+19.19); return fract((p.x+p.y)*p.x); }
void main(){ vec4 c=texture(u_c,v_uv); vec3 d=texture(u_d,v_uv).rgb;
  vec3 b=vec3(0.); for(int y=-1;y<=1;y++) for(int x=-1;x<=1;x++){ float w=(x==0?2.:1.)*(y==0?2.:1.); b+=texture(u_c,v_uv+u_px*vec2(x,y)).rgb*w; } b/=16.;
  vec3 col = c.rgb + (c.rgb-b)*u_sharp;            // u_sharp<0 làm mờ, >0 làm nét
  vec2 q=v_uv/u_px; float n=(hash(q)+hash(q+17.3)+hash(q+41.7)-1.5)*2.;
  col += n*u_noise;
  // chỗ tóc che trên mặt nguồn (không vá được từ nửa đối xứng) → dùng thẳng da ảnh đích (không trộn mờ)
  vec3 oc=texture(u_o,v_uv).rgb, o2=texture(u_o2,v_uv).rgb;
  float hole=max(smoothstep(0.03,0.3,oc.r), smoothstep(0.04,0.22,o2.r))*u_fill;
  float tsk=texture(u_mz,v_uv).r;
  float a=c.a*(1.-clamp(hole*1.5,0.,1.)*tsk);
  o=vec4(mix(d, clamp(col,0.,1.), a), 1.); }`,
  down: H + `uniform sampler2D u_t; uniform vec2 u_px; // texel của ảnh lớn hơn
void main(){ o=0.25*(texture(u_t,v_uv+u_px*vec2(-1.,-1.))+texture(u_t,v_uv+u_px*vec2(1.,-1.))+texture(u_t,v_uv+u_px*vec2(-1.,1.))+texture(u_t,v_uv+u_px*vec2(1.,1.))); }`,
  collapse: H + `uniform sampler2D u_a0; uniform sampler2D u_a1; uniform sampler2D u_b0; uniform sampler2D u_b1; uniform sampler2D u_m; uniform sampler2D u_r1; uniform float u_top;
void main(){ float m=texture(u_m,v_uv).r; vec3 a0=texture(u_a0,v_uv).rgb, b0=texture(u_b0,v_uv).rgb;
  if(u_top>0.5){ o=vec4(mix(b0,a0,m),1.); return; }
  vec3 la=a0-texture(u_a1,v_uv).rgb, lb=b0-texture(u_b1,v_uv).rgb;
  o=vec4(clamp(mix(lb,la,m)+texture(u_r1,v_uv).rgb,0.,1.),1.); }`,
  feather: H + `uniform sampler2D u_a; uniform sampler2D u_d; uniform sampler2D u_m; void main(){ float m=texture(u_m,v_uv).r; o=vec4(mix(texture(u_d,v_uv).rgb, texture(u_a,v_uv).rgb, m),1.); }`,
  // lõi mặt: chỉ dùng mặt nguồn đã chỉnh (không trộn tần số thấp của ảnh đích → không bóng ma)
  core: H + `uniform sampler2D u_r; uniform sampler2D u_a; uniform sampler2D u_m;
void main(){ float m=texture(u_m,v_uv).r; float k=smoothstep(0.88,0.995,m); o=vec4(mix(texture(u_r,v_uv).rgb, texture(u_a,v_uv).rgb, k),1.); }`,
  output: H + `uniform sampler2D u_d; uniform sampler2D u_r; uniform vec4 u_crop; uniform float u_has;
void main(){ vec3 c=texture(u_d,v_uv).rgb; if(u_has>0.5){ vec2 q=(v_uv-u_crop.xy)/u_crop.zw; if(q.x>=0.&&q.y>=0.&&q.x<=1.&&q.y<=1.) c=texture(u_r,q).rgb; } o=vec4(c,1.); }`,
};

function srgbToLab(r, g, b) {
  const lin = c => (c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4));
  const R = lin(r / 255), G = lin(g / 255), B = lin(b / 255);
  const X = (0.4124*R + 0.3576*G + 0.1805*B) / 0.95047, Y = 0.2126*R + 0.7152*G + 0.0722*B, Z = (0.0193*R + 0.1192*G + 0.9505*B) / 1.08883;
  const f = t => (t > 0.008856 ? Math.cbrt(t) : 7.787 * t + 16 / 116);
  const fx = f(X), fy = f(Y), fz = f(Z);
  return [116*fy - 16, 500*(fx - fy), 200*(fy - fz)];
}

/** Thống kê LAB bền vững (loại điểm ngoại lai như râu, tóc, bóng gắt) trên vùng mặt nạ */
function labStats(px, thresh = 0.5) {
  const L = [], M = [];
  for (let i = 0; i < px.length; i += 4) { const m = px[i+3] / 255; if (m < thresh) continue; L.push(srgbToLab(px[i], px[i+1], px[i+2])); M.push(m); }
  if (L.length < 12) return null;
  let mean = [0,0,0], std = [1,1,1], n = 0;
  for (let it = 0; it < 3; it++) {
    const s = [0,0,0], s2 = [0,0,0]; let W = 0;
    for (let j = 0; j < L.length; j++) {
      const l = L[j];
      if (it) { let d2 = 0; for (let k = 0; k < 3; k++) d2 += ((l[k] - mean[k]) / std[k]) ** 2 * (k ? 1 : 0.5); if (d2 > 7.5) continue; }
      for (let k = 0; k < 3; k++) { s[k] += M[j]*l[k]; s2[k] += M[j]*l[k]*l[k]; } W += M[j];
    }
    if (W < 6) break;
    mean = s.map(v => v / W); std = s2.map((v, k) => Math.max(0.8, Math.sqrt(Math.max(0, v / W - mean[k]*mean[k])))); n = W;
  }
  // độ dốc a,b theo L (vùng tối của da thường "ấm"/đậm màu hơn) – hồi quy có trọng số trên tập đã lọc
  let cl = 0, ca = 0, cb = 0, Wt = 0;
  for (let j = 0; j < L.length; j++) { const l = L[j]; let d2 = 0; for (let k = 0; k < 3; k++) d2 += ((l[k] - mean[k]) / std[k]) ** 2 * (k ? 1 : 0.5); if (d2 > 7.5) continue;
    const dl = l[0] - mean[0]; cl += M[j] * dl * dl; ca += M[j] * dl * (l[1] - mean[1]); cb += M[j] * dl * (l[2] - mean[2]); Wt += M[j]; }
  const slope = cl > 1e-3 ? [0, Math.max(-0.6, Math.min(0.6, ca / cl)), Math.max(-0.6, Math.min(0.6, cb / cl))] : [0, 0, 0];
  return { mean, std, n, slope };
}

export const DEFAULTS = {
  proc: 1024,         // độ phân giải xử lý tối đa vùng mặt
  identity: 0.55,     // giữ tỉ lệ ngũ quan của mặt nguồn (0 = theo hẳn mặt đích)
  mirror: true,       // lấy nửa mặt đối xứng khi mặt nguồn quay nghiêng
  relight: 1.0,       // chuyển ánh sáng tần số thấp
  color: 1.0,         // khớp màu LAB
  grain: true,        // khớp độ nét/hạt
  blend: 'multiband', // 'multiband' | 'feather'
  feather: 1.0,
  keepMouth: true,    // giữ miệng (răng) của ảnh đích khi đang mở
  keepEyes: true,     // giữ mắt ảnh đích khi đang nhắm
  occlusion: 1,       // giữ tóc mái/vật che mặt (phát hiện theo màu da)
  statsEvery: 0,      // 0 = tính mỗi lần (ảnh); >0 = cache theo số khung (video)
};

export class FaceSwapper {
  constructor(canvas) {
    this.canvas = canvas || document.createElement('canvas');
    const gl = this.canvas.getContext('webgl2', { premultipliedAlpha: false, preserveDrawingBuffer: true, antialias: false, alpha: false });
    if (!gl) throw new Error('Trình duyệt không hỗ trợ WebGL2');
    this.gl = gl;
    this.progs = {};
    for (const k in FS) this.progs[k] = this._prog(k === 'warp' ? VS_WARP : VS_QUAD, FS[k]);
    this.fbos = new Map();
    this.quad = this._quadVAO();
    this.warpBuf = gl.createBuffer(); this.warpVAO = this._warpVAO();
    this.srcTex = this._tex(); this.dstTex = this._tex();
    this.maskCanvas = document.createElement('canvas');
    this.mR = document.createElement('canvas'); this.mG = document.createElement('canvas'); this.mB = document.createElement('canvas'); this.mZ = document.createElement('canvas'); this.mT = document.createElement('canvas');
    this.frame = 0; this.stats = null; this.grainP = null;
    this.tris = triangles();
  }

  _prog(vs, fs) {
    const gl = this.gl, p = gl.createProgram();
    for (const [type, src] of [[gl.VERTEX_SHADER, vs], [gl.FRAGMENT_SHADER, fs]]) {
      const s = gl.createShader(type); gl.shaderSource(s, src); gl.compileShader(s);
      if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error('Shader: ' + gl.getShaderInfoLog(s));
      gl.attachShader(p, s);
    }
    gl.bindAttribLocation(p, 0, 'a_q'); gl.bindAttribLocation(p, 0, 'a_pos');
    gl.bindAttribLocation(p, 1, 'a_uv'); gl.bindAttribLocation(p, 2, 'a_uvm'); gl.bindAttribLocation(p, 3, 'a_w');
    gl.linkProgram(p);
    if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error('Link: ' + gl.getProgramInfoLog(p));
    const u = {}; const n = gl.getProgramParameter(p, gl.ACTIVE_UNIFORMS);
    for (let i = 0; i < n; i++) { const name = gl.getActiveUniform(p, i).name; u[name] = gl.getUniformLocation(p, name); }
    return { p, u };
  }
  _quadVAO() {
    const gl = this.gl, vao = gl.createVertexArray(); gl.bindVertexArray(vao);
    const b = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, b);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([0,0, 1,0, 0,1, 0,1, 1,0, 1,1]), gl.STATIC_DRAW);
    gl.enableVertexAttribArray(0); gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
    gl.bindVertexArray(null); return vao;
  }
  _warpVAO() {
    const gl = this.gl, vao = gl.createVertexArray(); gl.bindVertexArray(vao);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.warpBuf);
    const st = 9 * 4;
    gl.enableVertexAttribArray(0); gl.vertexAttribPointer(0, 3, gl.FLOAT, false, st, 0);
    gl.enableVertexAttribArray(1); gl.vertexAttribPointer(1, 2, gl.FLOAT, false, st, 12);
    gl.enableVertexAttribArray(2); gl.vertexAttribPointer(2, 2, gl.FLOAT, false, st, 20);
    gl.enableVertexAttribArray(3); gl.vertexAttribPointer(3, 2, gl.FLOAT, false, st, 28);
    gl.bindVertexArray(null); return vao;
  }
  _tex() {
    const gl = this.gl, t = gl.createTexture(); gl.bindTexture(gl.TEXTURE_2D, t);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR); gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE); gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    return t;
  }
  _upload(tex, source) {
    const gl = this.gl; gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false); gl.pixelStorei(gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL, false);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, source);
  }
  _fbo(name, w, h, depth = false) {
    w = Math.max(1, Math.round(w)); h = Math.max(1, Math.round(h));
    let f = this.fbos.get(name);
    if (f && f.w === w && f.h === h) return f;
    const gl = this.gl;
    if (f) { gl.deleteFramebuffer(f.fb); gl.deleteTexture(f.tex); f.rb && gl.deleteRenderbuffer(f.rb); }
    const tex = this._tex(); gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, w, h, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
    const fb = gl.createFramebuffer(); gl.bindFramebuffer(gl.FRAMEBUFFER, fb);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0);
    let rb = null;
    if (depth) { rb = gl.createRenderbuffer(); gl.bindRenderbuffer(gl.RENDERBUFFER, rb); gl.renderbufferStorage(gl.RENDERBUFFER, gl.DEPTH_COMPONENT16, w, h); gl.framebufferRenderbuffer(gl.FRAMEBUFFER, gl.DEPTH_ATTACHMENT, gl.RENDERBUFFER, rb); }
    f = { fb, tex, rb, w, h }; this.fbos.set(name, f); return f;
  }
  _pass(prog, target, uniforms = {}, textures = {}, flip = 0) {
    const gl = this.gl, P = this.progs[prog];
    gl.useProgram(P.p);
    if (target) { gl.bindFramebuffer(gl.FRAMEBUFFER, target.fb); gl.viewport(0, 0, target.w, target.h); }
    else { gl.bindFramebuffer(gl.FRAMEBUFFER, null); gl.viewport(0, 0, this.canvas.width, this.canvas.height); }
    let unit = 0;
    for (const k in textures) { gl.activeTexture(gl.TEXTURE0 + unit); gl.bindTexture(gl.TEXTURE_2D, textures[k]); if (P.u[k]) gl.uniform1i(P.u[k], unit); unit++; }
    for (const k in uniforms) {
      const loc = P.u[k]; if (!loc) continue; const v = uniforms[k];
      if (typeof v === 'number') gl.uniform1f(loc, v);
      else if (v.length === 2) gl.uniform2fv(loc, v); else if (v.length === 3) gl.uniform3fv(loc, v); else gl.uniform4fv(loc, v);
    }
    if (P.u.u_flip) gl.uniform1f(P.u.u_flip, flip);
    gl.bindVertexArray(this.quad); gl.drawArrays(gl.TRIANGLES, 0, 6); gl.bindVertexArray(null);
  }
  _blur(src, tmpName, dstName, w, h, sigma) {
    const t = this._fbo(tmpName, w, h), d = this._fbo(dstName, w, h);
    this._pass('blur', t, { u_dir: [1 / src.w, 0], u_sigma: Math.max(0.3, sigma * src.w / w) }, { u_t: src.tex });
    this._pass('blur', d, { u_dir: [0, 1 / h], u_sigma: Math.max(0.3, sigma) }, { u_t: t.tex });
    return d;
  }
  _read(f) {
    const gl = this.gl, px = new Uint8Array(f.w * f.h * 4);
    gl.bindFramebuffer(gl.FRAMEBUFFER, f.fb); gl.readPixels(0, 0, f.w, f.h, gl.RGBA, gl.UNSIGNED_BYTE, px);
    return px;
  }

  /** Ảnh nguồn (canvas/img/bitmap) + landmark (pixel) của mặt nguồn */
  setSource(source, lm, w, h) {
    this._upload(this.srcTex, source);
    this.src = { lm, w, h };
    this.srcVis = vertexVisibility(lm);
    this.srcMirror = mirrorWeights(this.srcVis);
    // độ tin cậy từng đỉnh: nhìn thấy rõ (trực tiếp hoặc qua nửa đối xứng) → 1; quay khuất → 0
    const raw = new Float32Array(N), mir = new Float32Array(N);
    for (let i = 0; i < N; i++) {
      const v = this.srcVis[i], vm = this.srcVis[MIRROR[i]], w = this.srcMirror[i];
      raw[i] = smooth(0.04, 0.3, v); mir[i] = smooth(0.04, 0.3, v * (1 - w) + vm * w);
    }
    this.srcQ = { raw, mir };
    this.srcYaw = yawOf(lm);
    this.stats = null; this.frame = 0;
  }

  /** Điểm đích: viền theo mặt đích, ngũ quan pha theo tỉ lệ mặt nguồn (đã căn tư thế 3D) */
  _targetPoints(dlm, identity) {
    const slm = this.src.lm, A = [], B = [], W = [];
    const inner = innerWeights();
    for (let i = 0; i < N; i++) { A.push([slm[i].x, slm[i].y, slm[i].z]); B.push([dlm[i].x, dlm[i].y, dlm[i].z]); W.push(0.3 + inner[i]); }
    const h = horn(A, B, W);
    const out = new Array(N);
    for (let i = 0; i < N; i++) {
      const q = h.apply(A[i]), a = identity * inner[i];
      out[i] = { x: B[i][0] + (q[0] - B[i][0]) * a, y: B[i][1] + (q[1] - B[i][1]) * a, z: B[i][2] + (q[2] - B[i][2]) * a };
    }
    return out;
  }

  _drawMasks(pts, dlm, crop, mw, mh, faceW, opts) {
    const sx = mw / crop.w, sy = mh / crop.h;
    const P = i => [(pts[i].x - crop.x) * sx, (pts[i].y - crop.y) * sy];
    const D = i => [(dlm[i].x - crop.x) * sx, (dlm[i].y - crop.y) * sy];
    const fw = faceW * sx;
    const path = (ctx, poly, close = true) => { ctx.beginPath(); poly.forEach((p, i) => (i ? ctx.lineTo(p[0], p[1]) : ctx.moveTo(p[0], p[1]))); if (close) ctx.closePath(); };
    // bao lồi từ viền mặt (khung theo đường mắt → chống nghiêng), kéo phần trán xuống để tránh chân tóc
    const hullOf = F => {
      const eR = F(33), eL = F(263), ec = [(eR[0] + eL[0]) / 2, (eR[1] + eL[1]) / 2];
      const chin = F(152); let up = [ec[0] - chin[0], ec[1] - chin[1]]; const ul = Math.hypot(...up) || 1; up = [up[0] / ul, up[1] / ul];
      const ctr = F(1);
      return convexHull(IDX.oval.map(i => {
        const p = F(i); const t = ((p[0] - ec[0]) * up[0] + (p[1] - ec[1]) * up[1]) / ul;
        const k = t > 0 ? 0.05 + 0.17 * Math.min(1, t / 0.9) : 0.02;
        return [p[0] + (ctr[0] - p[0]) * k, p[1] + (ctr[1] - p[1]) * k];
      }));
    };
    const skin = (ctx, F, col) => {
      const hull = hullOf(F);
      ctx.fillStyle = '#000'; ctx.fillRect(0, 0, mw, mh);
      ctx.fillStyle = col; path(ctx, hull); ctx.fill();
      ctx.strokeStyle = '#000'; ctx.fillStyle = '#000'; ctx.lineJoin = 'round'; ctx.lineCap = 'round';
      ctx.lineWidth = 2 * 0.07 * fw; path(ctx, hull); ctx.stroke();
      for (const idx of [IDX.leftEye, IDX.rightEye, IDX.lipsOuter]) { path(ctx, idx.map(F)); ctx.lineWidth = 0.07 * fw; ctx.fill(); ctx.stroke(); }
      for (const idx of [IDX.leftBrow, IDX.rightBrow]) { path(ctx, idx.map(F)); ctx.lineWidth = 0.08 * fw; ctx.fill(); ctx.stroke(); }
      for (const i of [98, 327]) { const p = F(i); ctx.beginPath(); ctx.arc(p[0], p[1], 0.05 * fw, 0, 7); ctx.fill(); } // cánh mũi
    };
    for (const c of [this.mR, this.mG, this.mB]) { c.width = mw; c.height = mh; }
    const r = this.mR.getContext('2d');
    const hull = hullOf(P);
    r.fillStyle = '#000'; r.fillRect(0, 0, mw, mh);
    r.fillStyle = '#f00'; path(r, hull); r.fill();
    r.strokeStyle = '#000'; r.lineJoin = 'round'; r.lineWidth = 2 * 0.03 * fw; path(r, hull); r.stroke();
    // luôn phủ kín lông mày của ảnh đích (tránh "hai cặp lông mày")
    r.fillStyle = '#f00'; r.strokeStyle = '#f00'; r.lineCap = 'round'; r.lineWidth = 0.07 * fw;
    for (const idx of [IDX.leftBrow, IDX.rightBrow]) { path(r, idx.map(D)); r.fill(); r.stroke(); }
    skin(this.mG.getContext('2d'), P, '#0f0');
    skin(this.mB.getContext('2d'), D, '#00f');
    // giữ miệng (răng) / mắt nhắm của ảnh đích
    const o = openness(dlm), os = openness(this.src.lm);
    r.fillStyle = '#000';
    if (opts.keepMouth && o.mouth > 0.06) { path(r, IDX.lipsInner.map(D)); r.fill(); }
    if (opts.keepEyes) {
      if (o.eyeR < 0.12 && os.eyeR > 0.18) { path(r, IDX.rightEye.map(D)); r.fill(); }
      if (o.eyeL < 0.12 && os.eyeL > 0.18) { path(r, IDX.leftEye.map(D)); r.fill(); }
    }
    // vùng được phép phát hiện che khuất: cả mặt trừ ngũ quan (nới rộng)
    { const z = this.mZ; z.width = mw; z.height = mh; const c = z.getContext('2d');
      c.fillStyle = '#000'; c.fillRect(0, 0, mw, mh); c.fillStyle = '#f00'; path(c, hull); c.fill();
      c.fillStyle = '#000'; c.strokeStyle = '#000'; c.lineJoin = 'round'; c.lineWidth = 0.12 * fw;
      for (const F of [P, D]) { c.lineWidth = 0.12 * fw;
        for (const idx of [IDX.leftEye, IDX.rightEye, IDX.lipsOuter, IDX.leftBrow, IDX.rightBrow]) { path(c, idx.map(F)); c.fill(); c.stroke(); }
        for (const i of [98, 327, 4]) { const p = F(i); c.beginPath(); c.arc(p[0], p[1], 0.08 * fw, 0, 7); c.fill(); }
      }
      // vùng an toàn để lấp bằng da ảnh đích: tránh xa ngũ quan của cả hai ảnh
      { const t = this.mT; t.width = mw; t.height = mh; const tc = t.getContext('2d');
        tc.fillStyle = '#000'; tc.fillRect(0, 0, mw, mh); tc.fillStyle = '#f00'; path(tc, hull); tc.fill();
        tc.fillStyle = '#000'; tc.strokeStyle = '#000'; tc.lineJoin = 'round';
        for (const F of [P, D]) {
          tc.lineWidth = 0.24 * fw; for (const idx of [IDX.leftEye, IDX.rightEye, IDX.leftBrow, IDX.rightBrow]) { path(tc, idx.map(F)); tc.fill(); tc.stroke(); }
          tc.lineWidth = 0.12 * fw; path(tc, IDX.lipsOuter.map(F)); tc.fill(); tc.stroke();
          for (const i of [98, 327, 4]) { const q = F(i); tc.beginPath(); tc.arc(q[0], q[1], 0.08 * fw, 0, 7); tc.fill(); }
        } }
      // lõi da để đo thống kê màu (thu nhỏ về giữa mặt → ít tóc/nền lẫn vào): G = nguồn, B = đích
      c.globalCompositeOperation = 'lighter';
      for (const [F, col] of [[P, '#0f0'], [D, '#00f']]) {
        const ctr = F(1), hh = hullOf(F).map(p => [ctr[0] + (p[0] - ctr[0]) * 0.72, ctr[1] + (p[1] - ctr[1]) * 0.72]);
        const t = document.createElement('canvas'); t.width = mw; t.height = mh; const tc = t.getContext('2d');
        tc.fillStyle = col; path(tc, hh); tc.fill();
        tc.globalCompositeOperation = 'destination-out'; tc.fillStyle = '#000'; tc.strokeStyle = '#000'; tc.lineJoin = 'round'; tc.lineCap = 'round'; tc.lineWidth = 0.08 * fw;
        for (const idx of [IDX.leftEye, IDX.rightEye, IDX.lipsOuter, IDX.leftBrow, IDX.rightBrow]) { path(tc, idx.map(F)); tc.fill(); tc.stroke(); }
        for (const i of [98, 327]) { const p = F(i); tc.beginPath(); tc.arc(p[0], p[1], 0.06 * fw, 0, 7); tc.fill(); }
        c.drawImage(t, 0, 0);
      }
      c.globalCompositeOperation = 'source-over'; }
    const m = this.maskCanvas; m.width = mw; m.height = mh;
    const mc = m.getContext('2d'); mc.globalCompositeOperation = 'source-over'; mc.drawImage(this.mR, 0, 0);
    mc.globalCompositeOperation = 'lighter'; mc.drawImage(this.mG, 0, 0); mc.drawImage(this.mB, 0, 0); mc.globalCompositeOperation = 'source-over';
    // khung mặt cho vùng trán (toạ độ uv của crop)
    const eR = P(33), eL = P(263), ec = [(eR[0] + eL[0]) / 2 / mw, (eR[1] + eL[1]) / 2 / mh], ch = P(152);
    const ux = ec[0] - ch[0] / mw, uy = ec[1] - ch[1] / mh, ul2 = ux * ux + uy * uy || 1;
    this._zone = { ec, up: [ux / ul2, uy / ul2] };
    return m;
  }

  /**
   * Ghép mặt nguồn lên ảnh/khung hình đích. Vẽ kết quả đầy đủ lên this.canvas.
   * dst: canvas/video/img; dlm: landmark đích (pixel) hoặc null (chỉ hiện ảnh gốc)
   */
  render(dst, dw, dh, dlm, options = {}) {
    const opts = { ...DEFAULTS, ...options };
    const gl = this.gl;
    if (this.canvas.width !== dw || this.canvas.height !== dh) { this.canvas.width = dw; this.canvas.height = dh; }
    if (opts.upload !== false) this._upload(this.dstTex, dst);
    if (!dlm || !this.src) { this._pass('output', null, { u_has: 0, u_crop: [0,0,1,1] }, { u_d: this.dstTex, u_r: this.dstTex }, 1); return null; }
    this.frame++;

    const pts = this._targetPoints(dlm, opts.identity);
    // vùng cắt quanh mặt (xử lý độc lập độ phân giải ảnh gốc)
    const fb = bbox(pts, IDX.oval), fbd = bbox(dlm, IDX.oval);
    const x0 = Math.min(fb.x, fbd.x), y0 = Math.min(fb.y, fbd.y), x1 = Math.max(fb.x + fb.w, fbd.x + fbd.w), y1 = Math.max(fb.y + fb.h, fbd.y + fbd.h);
    const faceW = Math.max(x1 - x0, y1 - y0), mg = 0.35 * faceW;
    const crop = { x: Math.max(0, Math.floor(x0 - mg)), y: Math.max(0, Math.floor(y0 - mg)) };
    crop.w = Math.min(dw, Math.ceil(x1 + mg)) - crop.x; crop.h = Math.min(dh, Math.ceil(y1 + mg)) - crop.y;
    const sc = Math.min(1.5, opts.proc / Math.max(crop.w, crop.h));
    const pw = Math.max(16, Math.round(crop.w * sc)), ph = Math.max(16, Math.round(crop.h * sc));
    const faceP = faceW * sc;

    // 1) warp tam giác (affine từng mảnh) + lấy mẫu đối xứng khi mặt nguồn bị khuất
    const T = this.tris, sw = this.src.w, sh = this.src.h, slm = this.src.lm;
    let zmin = Infinity, zmax = -Infinity; for (let i = 0; i < N; i++) { zmin = Math.min(zmin, pts[i].z); zmax = Math.max(zmax, pts[i].z); }
    const zr = Math.max(1e-3, zmax - zmin);
    const data = new Float32Array(T.length * 9), Q = this.srcQ;
    for (let k = 0; k < T.length; k++) {
      const i = T[k], p = pts[i], s = slm[i], m = slm[MIRROR[i]], o = k * 9;
      data[o] = (p.x - crop.x) / crop.w; data[o+1] = (p.y - crop.y) / crop.h; data[o+2] = ((p.z - zmin) / zr) * 1.8 - 0.9;
      data[o+3] = s.x / sw; data[o+4] = s.y / sh; data[o+5] = m.x / sw; data[o+6] = m.y / sh;
      data[o+7] = opts.mirror ? this.srcMirror[i] : 0;
      data[o+8] = opts.pose === false ? 1 : (opts.mirror ? Q.mir[i] : Q.raw[i]);
    }
    const drawWarp = (name, d) => {
      const F = this._fbo(name, pw, ph, true);
      gl.bindFramebuffer(gl.FRAMEBUFFER, F.fb); gl.viewport(0, 0, pw, ph);
      gl.clearColor(0, 0, 0, 0); gl.clearDepth(1); gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
      gl.enable(gl.DEPTH_TEST); gl.depthFunc(gl.LESS);
      gl.useProgram(this.progs.warp.p); gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, this.srcTex); gl.uniform1i(this.progs.warp.u.u_src, 0);
      gl.bindBuffer(gl.ARRAY_BUFFER, this.warpBuf); gl.bufferData(gl.ARRAY_BUFFER, d, gl.DYNAMIC_DRAW);
      gl.bindVertexArray(this.warpVAO); gl.drawArrays(gl.TRIANGLES, 0, T.length); gl.bindVertexArray(null);
      gl.disable(gl.DEPTH_TEST);
      return F;
    };
    const useOcc = (opts.occlusion ?? 1) > 0;
    let WM = null;
    if (useOcc && opts.mirror) { // bản lấy hoàn toàn từ nửa mặt đối xứng: dùng để vá chỗ tóc che trên mặt nguồn
      const dm = data.slice(); const V = this.srcVis;
      for (let k = 0; k < T.length; k++) { const o = k * 9, i = T[k]; dm[o+7] = 1; dm[o+8] = smooth(0.04, 0.3, V[MIRROR[i]]); }
      WM = drawWarp('warpm', dm);
    }
    const W = drawWarp('warp', data);

    // 2) cắt ảnh đích
    const Dc = this._fbo('dcrop', pw, ph);
    this._pass('crop', Dc, { u_crop: [crop.x / dw, crop.y / dh, crop.w / dw, crop.h / dh] }, { u_t: this.dstTex });

    // 3) mặt nạ: R = vùng ghép (co vào), G = vùng da (không mắt/mày/môi)
    const mw = Math.max(16, Math.round(pw / 2)), mh = Math.max(16, Math.round(ph / 2));
    const mcan = this._drawMasks(pts, dlm, crop, mw, mh, faceW, opts);
    const Mraw = this._fbo('mraw', mw, mh);
    gl.bindTexture(gl.TEXTURE_2D, Mraw.tex); gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, gl.RGBA, gl.UNSIGNED_BYTE, mcan);
    const Mz = this._fbo('mz', mw, mh);
    gl.bindTexture(gl.TEXTURE_2D, Mz.tex); gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, gl.RGBA, gl.UNSIGNED_BYTE, this.mZ);
    const Mt = this._fbo('mt', mw, mh);
    gl.bindTexture(gl.TEXTURE_2D, Mt.tex); gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, gl.RGBA, gl.UNSIGNED_BYTE, this.mT);
    const featherSigma = (opts.blend === 'multiband' ? 0.018 : 0.035) * opts.feather * faceP / 2; // theo px của mask
    const Mf = this._blur(Mraw, 'mt', 'mf', mw, mh, Math.max(0.5, featherSigma));

    // 4) thống kê LAB bền vững của da (nguồn thô / đích) – cache theo khung cho video
    const need = !this.stats || !opts.statsEvery || (this.frame % opts.statsEvery === 0);
    const S1 = this._fbo('s1', 64, 64), S2 = this._fbo('s2', 64, 64);
    const ema = (old, nw) => (old && opts.statsEvery && nw) ? { mean: old.mean.map((v, i) => v * 0.5 + nw.mean[i] * 0.5), std: old.std.map((v, i) => v * 0.5 + nw.std[i] * 0.5), slope: (old.slope || nw.slope || [0, 0, 0]).map((v, i) => v * 0.5 + (nw.slope || old.slope || [0, 0, 0])[i] * 0.5) } : (nw || old);
    if (need) {
      this._pass('statsPack', S1, { u_useA: 1, u_ch: 1 }, { u_t: W.tex, u_m: Mz.tex });
      this._pass('statsPack', S2, { u_useA: 0, u_ch: 2 }, { u_t: Dc.tex, u_m: Mz.tex });
      const a = labStats(this._read(S1)), b = labStats(this._read(S2));
      if (a && b) this.stats = { raw: ema(this.stats?.raw, a), dst: ema(this.stats?.dst, b), rel: this.stats?.rel };
    }
    if (!this.stats) this.stats = { raw: { mean: [60, 10, 15], std: [10, 5, 5] }, dst: { mean: [60, 10, 15], std: [10, 5, 5] } };
    const st = this.stats;
    const SU = { u_sm: st.raw.mean, u_ss: st.raw.std, u_dm: st.dst.mean, u_ds: st.dst.std };

    // 5) che khuất vùng trán (tóc mái…) → mặt nạ cuối
    const Sp = this._fbo('sp', mw, mh);
    this._pass('sprep', Sp, SU, { u_w: W.tex, u_m: Mraw.tex });
    const Sl = this._blur(Sp, 'st1', 'sl', mw, mh, Math.max(1, 0.05 * faceP / 2));
    let SlM = Sl;
    if (WM) { const Spm = this._fbo('spm', mw, mh); this._pass('sprep', Spm, SU, { u_w: WM.tex, u_m: Mraw.tex }); SlM = this._blur(Spm, 'st2', 'slm', mw, mh, Math.max(1, 0.05 * faceP / 2)); }
    const Oc = this._fbo('occ', mw, mh);
    this._pass('occ', Oc, { ...SU, u_ec: this._zone.ec, u_up: this._zone.up }, { u_w: W.tex, u_d: Dc.tex, u_z: Mz.tex, u_sl: Sl.tex, u_wm: (WM || W).tex, u_slm: SlM.tex });
    const Ob = this._blur(Oc, 'ot', 'ob', mw, mh, Math.max(0.5, 0.012 * faceP / 2));
    const Mc = this._fbo('mc', mw, mh);
    const Ob2 = this._blur(Oc, 'ot2', 'ob2', mw, mh, Math.max(0.5, 0.04 * faceP / 2));
    this._pass('mcomb', Mc, { u_amt: opts.occlusion ?? 1, u_big: 0 }, { u_m: Mf.tex, u_o: Ob.tex, u_o2: Ob2.tex });

    // 6) ánh sáng tần số thấp: tỉ lệ độ sáng đã làm mờ, chuẩn hoá theo vùng da của mỗi ảnh
    let cur = W;
    if (useOcc) {
      const Wi = this._fbo('wi', pw, ph);
      this._pass('inpaint', Wi, { u_hasM: WM ? 1 : 0 }, { u_w: W.tex, u_s: Sl.tex, u_o: Ob.tex, u_wm: (WM || W).tex, u_sm2: SlM.tex });
      cur = Wi;
    }
    if (opts.relight > 0) {
      const iw = Math.max(8, Math.round(pw / 4)), ih = Math.max(8, Math.round(ph / 4));
      const Ip = this._fbo('ip', iw, ih);
      this._pass('illumPrep', Ip, SU, { u_d: Dc.tex, u_w: cur.tex, u_m: Mc.tex });
      const Ib = this._blur(Ip, 'it', 'ib', iw, ih, Math.max(1, (opts.relightScale ?? 0.08) * faceP / 4));
      const R1 = this._fbo('relit', pw, ph);
      const slp = x => (x && x.slope) ? [x.slope[1], x.slope[2]] : [0, 0];
      this._pass('relight', R1, { u_strength: opts.relight, u_sm: st.raw.mean, u_dm: st.dst.mean, u_ks: slp(st.raw), u_kt: slp(st.dst), u_lim: opts.relightLim ?? 30 }, { u_w: cur.tex, u_i: Ib.tex });
      cur = R1;
    }

    // 7) khớp màu LAB (trung bình/độ lệch chuẩn) chỉ trên vùng da (đã loại mắt, mày, môi, tóc)
    if (opts.color > 0) {
      if (need || !st.rel) {
        this._pass('statsPack', S1, { u_useA: 1, u_ch: 1 }, { u_t: cur.tex, u_m: Mz.tex });
        st.rel = ema(st.rel, labStats(this._read(S1))) || st.raw;
      }
      const C = this._fbo('lab', pw, ph);
      const sl = x => (x && x.slope) ? [x.slope[1], x.slope[2]] : [0, 0];
      this._pass('labXfer', C, { u_ms: st.rel.mean, u_ss: st.rel.std, u_mt: st.dst.mean, u_st: st.dst.std, u_amt: opts.color, u_ks: [0, 0], u_kt: [0, 0] }, { u_w: cur.tex });
      cur = C;
    }

    // 8) khớp độ nét / hạt nhiễu (đo trên vùng da)
    let sharp = 0, noise = 0;
    if (opts.grain) {
      if (!opts.statsEvery || !this.grainP || this.frame % (opts.statsEvery * 4) === 0) {
        const Ms = this._fbo('meas', pw, ph);
        this._pass('measure', Ms, {}, { u_a: cur.tex, u_d: Dc.tex, u_m: Mc.tex });
        this.grainP = hfMatch(this._read(Ms), pw, ph);
      }
      ({ sharp, noise } = this.grainP);
    }
    const A = this._fbo('A', pw, ph);
    this._pass('finalA', A, { u_px: [1 / pw, 1 / ph], u_sharp: sharp, u_noise: noise, u_seed: opts.statsEvery ? 0.37 : (this.frame % 97) * 0.013, u_fill: useOcc ? (opts.occlusion ?? 1) : 0 }, { u_c: cur.tex, u_d: Dc.tex, u_o: Ob.tex, u_o2: Ob2.tex, u_mz: Mt.tex });

    // 9) trộn: đa băng tần (Laplacian pyramid) hoặc viền mềm
    let Rres;
    if (opts.blend === 'multiband') {
      const levels = Math.max(2, Math.min(opts.levels || 6, Math.floor(Math.log2(Math.min(pw, ph) / 8))));
      const GA = [A], GB = [Dc], GM = [];
      GM[0] = this._fbo('m0', pw, ph); this._pass('copy', GM[0], {}, { u_t: Mc.tex });
      for (let l = 1; l <= levels; l++) {
        const w = Math.max(1, Math.ceil(GA[l-1].w / 2)), h = Math.max(1, Math.ceil(GA[l-1].h / 2));
        for (const [arr, nm] of [[GA, 'ga'], [GB, 'gb'], [GM, 'gm']]) {
          const f = this._fbo(nm + l, w, h); this._pass('down', f, { u_px: [1 / arr[l-1].w, 1 / arr[l-1].h] }, { u_t: arr[l-1].tex }); arr[l] = f;
        }
      }
      let Rn = this._fbo('r' + levels, GA[levels].w, GA[levels].h);
      this._pass('collapse', Rn, { u_top: 1 }, { u_a0: GA[levels].tex, u_a1: GA[levels].tex, u_b0: GB[levels].tex, u_b1: GB[levels].tex, u_m: GM[levels].tex, u_r1: GA[levels].tex });
      for (let l = levels - 1; l >= 0; l--) {
        const f = this._fbo('r' + l, GA[l].w, GA[l].h);
        this._pass('collapse', f, { u_top: 0 }, { u_a0: GA[l].tex, u_a1: GA[l+1].tex, u_b0: GB[l].tex, u_b1: GB[l+1].tex, u_m: GM[l].tex, u_r1: Rn.tex });
        Rn = f;
      }
      Rres = this._fbo('rcore', pw, ph);
      this._pass('core', Rres, {}, { u_r: Rn.tex, u_a: A.tex, u_m: Mc.tex });
    } else {
      Rres = this._fbo('rf', pw, ph);
      this._pass('feather', Rres, {}, { u_a: A.tex, u_d: Dc.tex, u_m: Mc.tex });
    }

    // 10) xuất toàn khung
    this._pass('output', null, { u_has: 1, u_crop: [crop.x / dw, crop.y / dh, crop.w / dw, crop.h / dh] }, { u_d: this.dstTex, u_r: Rres.tex }, 1);
    if (opts.debug && this.fbos.get(opts.debug)) { const f = this.fbos.get(opts.debug); this.canvas.width = f.w; this.canvas.height = f.h; this._pass('copy', null, {}, { u_t: f.tex }, 1); }
    return { crop, pts, sharp, noise, stats: this.stats, yawSrc: this.srcYaw, yawDst: yawOf(dlm) };
  }
}

function hfMatch(px, w, h) {
  // độ lệch chuẩn thành phần tần số cao (Y - trung bình 3x3) trên vùng da của nguồn (r) và đích (g)
  let sa = 0, sd = 0, n = 0;
  const Y = (x, y, c) => px[(y * w + x) * 4 + c];
  for (let y = 2; y < h - 2; y += 2) for (let x = 2; x < w - 2; x += 2) {
    if (px[(y * w + x) * 4 + 2] < 230) continue;
    let ma = 0, md = 0;
    for (let j = -1; j <= 1; j++) for (let i = -1; i <= 1; i++) { ma += Y(x+i, y+j, 0); md += Y(x+i, y+j, 1); }
    const ha = Y(x, y, 0) - ma / 9, hd = Y(x, y, 1) - md / 9;
    sa += ha * ha; sd += hd * hd; n++;
  }
  if (n < 50) return { sharp: 0, noise: 0, n };
  const a = Math.sqrt(sa / n) / 255, d = Math.sqrt(sd / n) / 255;
  let sharp = 0, noise = 0;
  if (a > d * 1.25) sharp = -Math.min(0.9, (a - d) / a * 1.2);           // nguồn nét hơn → làm mờ nhẹ
  else if (a < d * 0.85) {
    sharp = Math.min(0.8, (d - a) / d * 1.0);                              // nguồn mờ hơn → làm nét nhẹ
    noise = Math.min(0.035, Math.sqrt(Math.max(0, d * d - a * a * (1 + sharp) ** 2)) * 0.7); // thêm hạt
  }
  return { sharp, noise, a, d, n };
}

function convexHull(points) {
  const p = points.map(q => q.slice()).sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  const cross = (o, a, b) => (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
  const lo = [], up = [];
  for (const q of p) { while (lo.length >= 2 && cross(lo[lo.length - 2], lo[lo.length - 1], q) <= 0) lo.pop(); lo.push(q); }
  for (let i = p.length - 1; i >= 0; i--) { const q = p[i]; while (up.length >= 2 && cross(up[up.length - 2], up[up.length - 1], q) <= 0) up.pop(); up.push(q); }
  up.pop(); lo.pop(); return lo.concat(up);
}
