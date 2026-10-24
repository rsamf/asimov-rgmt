// Motion-matching locomotion controller: a plain-JS port of locomotion/controller.py,
// locomotion/player.py, locomotion/trajectory.py and the query side of
// locomotion/features.py. Quaternions are xyzw. No dependencies; runs in browsers and Node.

const LN2 = Math.log(2.0);
const POSE_DIM = 15;
const f32 = Math.fround;

// ---- geometry ------------------------------------------------------------------------
export function wrapAngle(a) {
  const p = 2 * Math.PI;
  return ((((a + Math.PI) % p) + p) % p) - Math.PI;   // Python % semantics (non-negative)
}

export function quatMul(a, b) {
  const [ax, ay, az, aw] = a, [bx, by, bz, bw] = b;
  return [
    aw * bx + ax * bw + ay * bz - az * by,
    aw * by - ax * bz + ay * bw + az * bx,
    aw * bz + ax * by - ay * bx + az * bw,
    aw * bw - ax * bx - ay * by - az * bz,
  ];
}

export function quatConj(q) { return [-q[0], -q[1], -q[2], q[3]]; }

export function yawFromQuat(q) {
  const [x, y, z, w] = q;
  return Math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z));
}

export function quatFromYaw(yaw) { const h = yaw / 2; return [0, 0, Math.sin(h), Math.cos(h)]; }

export function quatToRotvec(q) {
  let n = Math.max(Math.hypot(q[0], q[1], q[2], q[3]), 1e-12);
  let x = q[0] / n, y = q[1] / n, z = q[2] / n, w = q[3] / n;
  if (w < 0) { x = -x; y = -y; z = -z; w = -w; }
  const vn = Math.hypot(x, y, z);
  const angle = 2.0 * Math.atan2(vn, w);
  const s = vn < 1e-6 ? 2.0 : angle / Math.max(vn, 1e-12);
  return [x * s, y * s, z * s];
}

export function rotvecToQuat(r) {
  const angle = Math.hypot(r[0], r[1], r[2]);
  const half = angle / 2;
  const k = angle < 1e-6 ? 0.5 : Math.sin(half) / Math.max(angle, 1e-12);
  const q = [r[0] * k, r[1] * k, r[2] * k, Math.cos(half)];
  const n = Math.hypot(q[0], q[1], q[2], q[3]);
  return [q[0] / n, q[1] / n, q[2] / n, q[3] / n];
}

function rotate2(v, yaw) {
  const c = Math.cos(yaw), s = Math.sin(yaw);
  return [c * v[0] - s * v[1], s * v[0] + c * v[1]];
}

// ---- springs (Holden closed forms) -------------------------------------------------------
function halflifeToDamping(halflife, eps = 1e-5) { return 4.0 * LN2 / (halflife + eps); }

// scalar decay; returns [x, v]
function decay(x, v, halflife, dt) {
  const y = halflifeToDamping(halflife) / 2.0;
  const j1 = v + x * y;
  const e = Math.exp(-y * dt);
  return [e * (x + j1 * dt), e * (v - j1 * y * dt)];
}

function decayArr(x, v, halflife, dt) {
  const nx = new Array(x.length), nv = new Array(x.length);
  for (let i = 0; i < x.length; i++) { const r = decay(x[i], v[i], halflife, dt); nx[i] = r[0]; nv[i] = r[1]; }
  return [nx, nv];
}

// scalar spring velocity update; returns [x, v, a]
function springVel(x, v, a, vDes, halflife, dt) {
  const y = halflifeToDamping(halflife) / 2.0;
  const j0 = v - vDes;
  const j1 = a + j0 * y;
  const e = Math.exp(-y * dt);
  const xn = e * ((-j1) / (y * y) + (-j0 - j1 * dt) / y) + j1 / (y * y) + j0 / y + vDes * dt + x;
  const vn = e * (j0 + j1 * dt) + vDes;
  const an = e * (a - j1 * y * dt);
  return [xn, vn, an];
}

class TrajectoryModel {
  constructor(spring, fps, horizons) {
    this.vh = spring.velocity_halflife; this.rh = spring.rotation_halflife;
    this.dt = 1.0 / fps; this.horizons = horizons.map(Number);
    this.reset();
  }
  reset() { this.vel = [0, 0]; this.acc = [0, 0]; this.yawRate = 0; this.yawAcc = 0; }
  step(vDes, wDes) {
    for (let i = 0; i < 2; i++) {
      const r = springVel(0, this.vel[i], this.acc[i], vDes[i], this.vh, this.dt);
      this.vel[i] = r[1]; this.acc[i] = r[2];
    }
    const r = springVel(0, this.yawRate, this.yawAcc, wDes, this.rh, this.dt);
    this.yawRate = r[1]; this.yawAcc = r[2];
  }
  // world-frame displacements and relative yaws at the given times (s)
  predictDeltasAt(times, vDes, wDes) {
    const dp = [], dyaw = [];
    for (const t of times) {
      dp.push([0, 1].map(i => springVel(0, this.vel[i], this.acc[i], vDes[i], this.vh, t)[0]));
      dyaw.push(springVel(0, this.yawRate, this.yawAcc, wDes, this.rh, t)[0]);
    }
    return { dp, dyaw };
  }
  predictDeltas(vDes, wDes) { return this.predictDeltasAt(this.horizons.map(h => h * this.dt), vDes, wDes); }
  // == Python predict_world: future world positions and yaws, for drawing
  predictWorld(xy0, yaw0, vDes, wDes, times = null) {
    const { dp, dyaw } = times ? this.predictDeltasAt(times, vDes, wDes) : this.predictDeltas(vDes, wDes);
    return { xy: dp.map(d => [xy0[0] + d[0], xy0[1] + d[1]]), yaw: dyaw.map(a => yaw0 + a) };
  }
  // -> traj_raw (4H): all heading-frame positions, then all facing directions
  predictRaw(yaw0, vDes, wDes) {
    const H = this.horizons.length, out = new Array(4 * H);
    const { dp: dps, dyaw: dyaws } = this.predictDeltas(vDes, wDes);
    for (let k = 0; k < H; k++) {
      const dp = dps[k], dyaw = dyaws[k];
      const ph = rotate2(dp, -yaw0);
      out[2 * k] = ph[0]; out[2 * k + 1] = ph[1];
      out[2 * H + 2 * k] = Math.cos(dyaw); out[2 * H + 2 * k + 1] = Math.sin(dyaw);
    }
    return out;
  }
}

// ---- motion / feature database --------------------------------------------------------
export function makeMotion(json, motionF32, rawF32) {
  const N = json.n_frames, row = json.row, L = json.layout;
  const D = json.feat.dim;
  if (motionF32.length !== N * row) throw new Error(`motion.bin: ${motionF32.length} != ${N}x${row}`);
  if (rawF32.length !== N * D) throw new Error(`features.bin: ${rawF32.length} != ${N}x${D}`);
  const cfg = json.config;
  const horizons = cfg.features.horizons.map(Number);
  const hmax = Math.max(...horizons);
  const C = json.clip_start.length;
  const clipId = new Int32Array(N);
  for (let c = 0; c < C; c++) for (let i = json.clip_start[c]; i <= json.clip_end[c]; i++) clipId[i] = c;
  const clipEnd = Int32Array.from(json.clip_end);
  const validPos = new Int32Array(N).fill(-1);
  const vIdx = [];
  for (let i = 0; i < N; i++) if (i + hmax <= clipEnd[clipId[i]]) { validPos[i] = vIdx.length; vIdx.push(i); }
  if (!vIdx.length) throw new Error("FeatureDB: no valid frame");
  const M = vIdx.length;
  const validIdx = Int32Array.from(vIdx);
  const mean = Float32Array.from(json.feat.mean), scale = Float32Array.from(json.feat.scale),
    weight = Float32Array.from(json.feat.weight);
  const feats = new Float32Array(M * D);
  const roomValid = new Int32Array(M), clipValid = new Int32Array(M);
  for (let m = 0; m < M; m++) {
    const i = validIdx[m];
    for (let c = 0; c < D; c++)
      feats[m * D + c] = f32(f32(f32(rawF32[i * D + c] - mean[c]) / scale[c]) * weight[c]);
    roomValid[m] = clipEnd[clipId[i]] - i;
    clipValid[m] = clipId[i];
  }
  const speeds = {};
  for (const k of ["forward", "backward", "lateral", "turn_rate"]) {
    const v = cfg.speeds?.[k] ?? json.speeds?.[k];
    if (v == null) throw new Error(`nominal speed '${k}' missing`);
    speeds[k] = Number(v);
  }
  const sl = (name) => L[name];
  return {
    json, cfg, N, row, D, data: motionF32, raw: rawF32, horizons, hmax,
    clipStart: Int32Array.from(json.clip_start), clipEnd, clipId, clipNames: json.clip_names,
    clipClass: json.clip_class, validIdx, validPos, feats, roomValid, clipValid,
    mean, scale, weight, speeds,
    off: { pos: sl("base_pos")[0], quat: sl("base_quat")[0], lin: sl("base_lin_vel")[0],
      ang: sl("base_ang_vel")[0], q: sl("joint_q")[0], qd: sl("joint_qd")[0] },
  };
}

export async function loadMotion(baseUrl) {
  const base = String(baseUrl).endsWith("/") ? String(baseUrl) : String(baseUrl) + "/";
  const get = async (n) => { const r = await fetch(base + n); if (!r.ok) throw new Error(`${n}: HTTP ${r.status}`); return r; };
  const json = await (await get("motion.json")).json();
  const [mb, fb] = await Promise.all([get("motion.bin").then(r => r.arrayBuffer()), get("features.bin").then(r => r.arrayBuffer())]);
  return makeMotion(json, new Float32Array(mb), new Float32Array(fb));
}

function isValid(m, i) { return i >= 0 && i < m.N && m.validPos[i] >= 0; }

// FeatureDB.query: brute force over all valid frames, one tight loop.
function query(m, poseRaw, trajRaw, currentFrame, currentNext, stayBias, jumpMargin, excludeWindow) {
  const D = m.D, q = new Float64Array(D);
  for (let c = 0; c < D; c++) {
    const r = c < POSE_DIM ? poseRaw[c] : f32(trajRaw[c - POSE_DIM]);
    q[c] = f32(f32(f32(r - m.mean[c]) / m.scale[c]) * m.weight[c]);
  }
  const feats = m.feats, M = m.validIdx.length, room = m.roomValid, clipV = m.clipValid, vIdx = m.validIdx;
  const minRoom = m.hmax + (jumpMargin | 0);
  const useEx = currentFrame != null && currentFrame >= 0 && currentFrame < m.N && excludeWindow > 0;
  const curClip = useEx ? m.clipId[currentFrame] : -1;
  const ew = excludeWindow | 0;
  const cont = (currentNext != null && isValid(m, currentNext)) ? m.validPos[currentNext] : -1;
  let best = Infinity, bestK = -1, any = Infinity, anyK = 0;
  for (let k = 0; k < M; k++) {
    const o = k * D;
    let d = 0;
    for (let c = 0; c < D; c++) { const e = feats[o + c] - q[c]; d += e * e; }
    if (d < any) { any = d; anyK = k; }
    if (k === cont) { d -= stayBias; }
    else if (room[k] < minRoom || (useEx && clipV[k] === curClip && Math.abs(vIdx[k] - currentFrame) <= ew)) continue;
    if (d < best) { best = d; bestK = k; }
  }
  if (bestK < 0 || !Number.isFinite(best)) bestK = anyK;
  return vIdx[bestK];
}

// ---- commands ---------------------------------------------------------------------------
const MOVE = { w: [1, 0], s: [-1, 0], a: [0, 1], d: [0, -1] };
const YAW = { q: 1, e: -1 };

export function fromKeys(keySet) {
  const keys = new Set([...keySet].map(k => String(k).toLowerCase()));
  let x = 0, y = 0, yaw = 0;
  for (const k of keys) {
    if (MOVE[k]) { x += MOVE[k][0]; y += MOVE[k][1]; }
    if (YAW[k] !== undefined) yaw += YAW[k];
  }
  const n = Math.hypot(x, y);
  if (n > 1.0) { x /= n; y /= n; }
  yaw = Math.max(-1, Math.min(1, yaw));
  return { move: [x + 0.0, y + 0.0], yaw: yaw + 0.0 };   // +0.0 turns -0 into 0
}

const cmdEq = (a, b) => a.move[0] === b.move[0] && a.move[1] === b.move[1] && a.yaw === b.yaw;
const ZERO_CMD = Object.freeze({ move: [0, 0], yaw: 0 });

// ---- player -----------------------------------------------------------------------------
function cloneFrame(f) {
  return { base_pos: f.base_pos.slice(), base_quat: f.base_quat.slice(), base_lin_vel: f.base_lin_vel.slice(),
    base_ang_vel: f.base_ang_vel.slice(), joint_q: f.joint_q.slice(), joint_qd: f.joint_qd.slice() };
}

class Player {
  constructor(m, halflife, fps) {
    this.m = m; this.hl = halflife; this.dt = 1.0 / fps;
    this.frame = 0; this.emitCurrent = true; this.t = [0, 0]; this.theta = 0; this.last = null;
    this.zeroOffsets();
  }
  zeroOffsets() {
    const z = n => new Array(n).fill(0);
    this.off = { q: z(23), qd: z(23), z: 0, vz: 0, rot: z(3), w: z(3) };
  }
  get clipEnd() { return this.m.clipEnd[this.m.clipId[this.frame]]; }
  get nextFrame() { return this.emitCurrent ? this.frame : Math.min(this.frame + 1, this.clipEnd); }
  get worldXY() { return [this.last.base_pos[0], this.last.base_pos[1]]; }
  get worldYaw() { return yawFromQuat(this.last.base_quat); }

  clipRoot(frame) {
    const d = this.m.data, o = frame * this.m.row, of = this.m.off;
    return [[d[o + of.pos], d[o + of.pos + 1]],
      yawFromQuat([d[o + of.quat], d[o + of.quat + 1], d[o + of.quat + 2], d[o + of.quat + 3]])];
  }
  setAnchor(frame, xy, yaw) {
    const [cxy, cyaw] = this.clipRoot(frame);
    this.theta = wrapAngle(yaw - cyaw);
    const r = rotate2(cxy, this.theta);
    this.t = [xy[0] - r[0], xy[1] - r[1]];
  }
  // corpus frame -> world through the anchor (no blending)
  raw(frame) {
    const d = this.m.data, o = frame * this.m.row, of = this.m.off;
    const c = Math.cos(this.theta), s = Math.sin(this.theta);
    const px = d[o + of.pos], py = d[o + of.pos + 1];
    const v3 = (a) => { const x = d[o + a], y = d[o + a + 1]; return [c * x - s * y, s * x + c * y, d[o + a + 2]]; };
    const sl = (a, n) => Array.from(d.subarray(o + a, o + a + n));
    return {
      base_pos: [c * px - s * py + this.t[0], s * px + c * py + this.t[1], d[o + of.pos + 2]],
      base_quat: quatMul(quatFromYaw(this.theta), sl(of.quat, 4)),
      base_lin_vel: v3(of.lin), base_ang_vel: v3(of.ang),
      joint_q: sl(of.q, 23), joint_qd: sl(of.qd, 23),
    };
  }
  reset(frame, xy, yaw) {
    this.frame = frame | 0;
    this.setAnchor(this.frame, xy, yaw);
    this.zeroOffsets();
    this.emitCurrent = true;
    this.last = this.raw(this.frame);
  }
  switch(frame) {
    const xy = this.worldXY, yaw = this.worldYaw, last = this.last;
    this.frame = frame | 0;
    this.setAnchor(this.frame, xy, yaw);
    const r = this.raw(this.frame);
    this.off = {
      q: last.joint_q.map((v, i) => v - r.joint_q[i]),
      qd: last.joint_qd.map((v, i) => v - r.joint_qd[i]),
      z: last.base_pos[2] - r.base_pos[2],
      vz: last.base_lin_vel[2] - r.base_lin_vel[2],
      rot: quatToRotvec(quatMul(last.base_quat, quatConj(r.base_quat))),
      w: last.base_ang_vel.map((v, i) => v - r.base_ang_vel[i]),
    };
    this.emitCurrent = true;
  }
  translate(dxy) {
    this.t = [this.t[0] + dxy[0], this.t[1] + dxy[1]];
    if (this.last) { this.last.base_pos[0] += dxy[0]; this.last.base_pos[1] += dxy[1]; }
  }
  decayed(o) {
    const [q, qd] = decayArr(o.q, o.qd, this.hl, this.dt);
    const [z, vz] = decay(o.z, o.vz, this.hl, this.dt);
    const [rot, w] = decayArr(o.rot, o.w, this.hl, this.dt);
    return { q, qd, z, vz, rot, w };
  }
  static blend(f, o) {        // f is a fresh raw frame; modified in place
    for (let i = 0; i < 23; i++) { f.joint_q[i] += o.q[i]; f.joint_qd[i] += o.qd[i]; }
    f.base_pos[2] += o.z;
    f.base_lin_vel[2] += o.vz;
    f.base_quat = quatMul(rotvecToQuat(o.rot), f.base_quat);
    for (let i = 0; i < 3; i++) f.base_ang_vel[i] += o.w[i];
    return f;
  }
  advance() {
    if (this.emitCurrent) this.emitCurrent = false;
    else this.frame = Math.min(this.frame + 1, this.clipEnd);
    const out = Player.blend(this.raw(this.frame), this.off);
    this.last = out;
    this.off = this.decayed(this.off);   // this tick used the undecayed offset
    return cloneFrame(out);
  }
  preview(k) {                  // == Python preview(k, blended=True)
    const start = this.nextFrame, end = this.clipEnd, res = [];
    let o = this.off;
    for (let i = 0; i < k; i++) {
      res.push(Player.blend(this.raw(Math.min(start + i, end)), o));
      o = this.decayed(o);
    }
    return res;
  }
}

// ---- controller -----------------------------------------------------------------------
export class Controller {
  constructor(motion) {
    this.m = motion;
    this.cfg = motion.cfg;
    this.speeds = motion.speeds;
    this.traj = new TrajectoryModel(this.cfg.spring, this.cfg.fps, motion.horizons);
    this.player = new Player(motion, this.cfg.blend.halflife, this.cfg.fps);
    this.lastCmd = ZERO_CMD;
    this.sinceSearch = 0;
    this.lastSearched = false;
  }
  firstFrameOfClass(cls) {
    for (let i = 0; i < this.m.clipClass.length; i++) if (this.m.clipClass[i] === cls) return this.m.clipStart[i];
    return 0;
  }
  reset(xy = [0, 0], yaw = 0, frame = null) {
    const f = frame == null ? this.firstFrameOfClass("idle") : frame | 0;
    this.player.reset(f, xy, yaw);
    this.traj.reset();
    this.lastCmd = ZERO_CMD;
    this.sinceSearch = this.cfg.search.interval;   // search on the first step
  }
  desired(cmd, yaw) {
    const [mx, my] = cmd.move;
    const vx = mx * (mx >= 0 ? this.speeds.forward : this.speeds.backward);
    const vy = my * this.speeds.lateral;
    return [rotate2([vx, vy], yaw), cmd.yaw * this.speeds.turn_rate];
  }
  step(cmd) {
    cmd = { move: [Number(cmd.move[0]), Number(cmd.move[1])], yaw: Number(cmd.yaw) };
    const p = this.player, s = this.cfg.search;
    const yaw0 = p.worldYaw;
    const [vDes, wDes] = this.desired(cmd, yaw0);
    const need = this.sinceSearch >= s.interval || (s.on_command_change && !cmdEq(cmd, this.lastCmd))
      || !isValid(this.m, p.nextFrame);
    this.lastSearched = need;
    if (need) {
      const D = this.m.D, o = p.frame * D;
      const pose = this.m.raw.subarray(o, o + POSE_DIM);
      const tr = this.traj.predictRaw(yaw0, vDes, wDes);
      const j = query(this.m, pose, tr, p.frame, p.nextFrame, s.stay_bias, s.jump_margin, s.exclude_window);
      if (j !== p.nextFrame) p.switch(j);
      this.sinceSearch = 0;
    }
    const out = p.advance();
    this.traj.step(vDes, wDes);
    this.sinceSearch += 1;
    this.lastCmd = cmd;
    this._vDes = vDes; this._wDes = wDes;
    return out;
  }
  // Predicted trajectory from the post-step state (Python _update_debug), for the
  // overlay: the horizon knots plus `nSamples` evenly spaced points for a smooth curve.
  trajectory(nSamples = 12) {
    const p = this.player, t = this.traj;
    const vDes = this._vDes ?? [0, 0], wDes = this._wDes ?? 0;
    const root = p.worldXY, yaw0 = p.worldYaw;
    const knots = t.predictWorld(root, yaw0, vDes, wDes);
    const tMax = Math.max(...t.horizons) * t.dt;
    const times = Array.from({ length: nSamples }, (_, i) => tMax * (i + 1) / nSamples);
    const samples = t.predictWorld(root, yaw0, vDes, wDes, times);
    return { root: [root[0], root[1]], yaw0, xy: knots.xy, yaw: knots.yaw, samples: samples.xy, vDes, wDes };
  }
  preview(k) { return this.player.preview(k); }
  translate(shift) { this.player.translate(shift); }
  get worldYaw() { return this.player.worldYaw; }
  get worldXY() { return this.player.worldXY; }
  // debug helpers
  get frame() { return this.player.frame; }
  get clipName() { return this.m.clipNames[this.m.clipId[this.player.frame]]; }
  get clipClass() { return this.m.clipClass[this.m.clipId[this.player.frame]]; }
}
