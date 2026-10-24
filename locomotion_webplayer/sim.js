// MuJoCo WASM wrapper: loads the Asimov MJCF + meshes into the Emscripten VFS,
// teleports the robot onto controller frames, builds the actor observation and
// checks the fall criteria (mirrors rgmt/env/track_env.py _build_obs/_fallen).
//
// Conventions: frames are xyzw quaternions; MuJoCo freejoint qpos is
// [x y z qw qx qy qz] and qvel is [linear world (3), angular BODY (3)].
import loadMujoco from 'https://cdn.jsdelivr.net/npm/@mujoco/mujoco@3.15.0/mujoco.js';

// ---- small quaternion helpers (xyzw) --------------------------------------
export function quatToMat(q) {            // row-major 3x3
  const [x, y, z, w] = q;
  return [
    1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
    2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
    2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y),
  ];
}
export function rotInv(R, v) {            // R^T v
  return [
    R[0] * v[0] + R[3] * v[1] + R[6] * v[2],
    R[1] * v[0] + R[4] * v[1] + R[7] * v[2],
    R[2] * v[0] + R[5] * v[1] + R[8] * v[2],
  ];
}
export function rot(R, v) {
  return [
    R[0] * v[0] + R[1] * v[1] + R[2] * v[2],
    R[3] * v[0] + R[4] * v[1] + R[5] * v[2],
    R[6] * v[0] + R[7] * v[1] + R[8] * v[2],
  ];
}
export function yawOfQuat(q) {
  const [x, y, z, w] = q;
  return Math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z));
}

async function fetchBytes(url, onBytes) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`fetch ${url}: HTTP ${r.status}`);
  if (!r.body || !onBytes) return new Uint8Array(await r.arrayBuffer());
  const reader = r.body.getReader();
  const chunks = []; let n = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value); n += value.length; onBytes(value.length);
  }
  const out = new Uint8Array(n); let o = 0;
  for (const c of chunks) { out.set(c, o); o += c.length; }
  return out;
}

export class Sim {
  static async load(base, plant, xmlName, onProgress = () => {}) {
    onProgress('Loading MuJoCo WASM…', 0);
    const mj = await loadMujoco();
    const xml = await (await fetch(base + xmlName)).text();
    const meshdir = (xml.match(/meshdir="([^"]*)"/) || [, ''])[1];
    const files = [...new Set([...xml.matchAll(/<mesh\b[^>]*\bfile="([^"]+)"/g)].map(m => m[1]))];
    // byte-level progress: HEAD sizes are not worth the round trips, use a rough total
    let got = 0; const est = 46e6;
    const vfs = new mj.MjVFS();
    const prefix = meshdir ? meshdir.replace(/\/$/, '') + '/' : '';
    let i = 0;
    await Promise.all(files.map(async f => {
      const bytes = await fetchBytes(base + prefix + f, n => {
        got += n; onProgress(`Loading meshes… ${(got / 1e6).toFixed(1)} MB`, Math.min(0.95, got / est));
      });
      vfs.addBuffer(prefix + f, bytes);
      i++;
    }));
    onProgress('Compiling model…', 0.97);
    const model = mj.MjModel.from_xml_string(xml, vfs);
    vfs.delete();
    return new Sim(mj, model, plant);
  }

  constructor(mj, model, plant) {
    this.mj = mj; this.model = model; this.plant = plant;
    this.data = new mj.MjData(model);
    this.nAct = plant.joint_names.length;
    this.qadr = plant.qpos_adr; this.vadr = plant.qvel_adr; this.cidx = plant.ctrl_idx;
    const fall = plant.fall || {};
    this.pelvis = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY.value, plant.pelvis_body || 'pelvis_link');
    this.headBody = fall.head_body ? mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY.value, fall.head_body) : -1;
    this.substeps = plant.substeps || 1;
    const ts = model.opt.timestep;
    console.log(`[sim] nq=${model.nq} nv=${model.nv} nu=${model.nu} ngeom=${model.ngeom} timestep=${ts} headBody=${this.headBody}`);
    if (model.nu < this.nAct) console.warn(`[sim] model has ${model.nu} actuators, plant expects ${this.nAct}`);
  }

  // views can detach when the WASM heap grows, so fetch them fresh
  get qpos() { return this.data.qpos; }
  get qvel() { return this.data.qvel; }

  rootPos() { const q = this.qpos; return [q[0], q[1], q[2]]; }
  rootQuat() { const q = this.qpos; return [q[4], q[5], q[6], q[3]]; }  // xyzw

  teleport(f) {
    const mj = this.mj, m = this.model, d = this.data;
    mj.mj_resetData(m, d);
    const qp = d.qpos, qv = d.qvel;
    const [x, y, z, w] = f.base_quat;
    qp[0] = f.base_pos[0]; qp[1] = f.base_pos[1]; qp[2] = f.base_pos[2];
    qp[3] = w; qp[4] = x; qp[5] = y; qp[6] = z;
    const R = quatToMat(f.base_quat);
    const wb = rotInv(R, f.base_ang_vel);
    qv[0] = f.base_lin_vel[0]; qv[1] = f.base_lin_vel[1]; qv[2] = f.base_lin_vel[2];
    qv[3] = wb[0]; qv[4] = wb[1]; qv[5] = wb[2];
    for (let i = 0; i < this.nAct; i++) { qp[this.qadr[i]] = f.joint_q[i]; qv[this.vadr[i]] = f.joint_qd[i]; }
    const c = d.ctrl;
    for (let i = 0; i < this.nAct; i++) c[this.cidx[i]] = f.joint_q[i];
    mj.mj_forward(m, d);
  }

  step(target) {
    const c = this.data.ctrl;
    for (let i = 0; i < this.nAct; i++) c[this.cidx[i]] = target[i];
    for (let s = 0; s < this.substeps; s++) this.mj.mj_step(this.model, this.data);
  }

  // obs (98) = [g_proj(3), omega(3), cos q(23), sin q(23), qd(23), prev_action(23)]
  buildObs(prevAction, out = new Float32Array(this.plant.obs_dim || 98)) {
    const qp = this.qpos, qv = this.qvel, n = this.nAct;
    const R = quatToMat(this.rootQuat());
    const g = rotInv(R, [0, 0, -1]);
    let om = [qv[3], qv[4], qv[5]];                 // body frame
    if (!this.plant.body_frame_ang_vel) om = rot(R, om);
    out[0] = g[0]; out[1] = g[1]; out[2] = g[2];
    out[3] = om[0]; out[4] = om[1]; out[5] = om[2];
    for (let i = 0; i < n; i++) {
      const q = qp[this.qadr[i]];
      out[6 + i] = Math.cos(q);
      out[6 + n + i] = Math.sin(q);
      out[6 + 2 * n + i] = qv[this.vadr[i]];
      out[6 + 3 * n + i] = prevAction[i];
    }
    return out;
  }

  fallen() {
    const f = this.plant.fall || {};
    // scripts/mujoco_sim2sim.py MjDriver.fallen: pelvis body z, pelvis R[2,2], head body z
    const xpos = this.data.xpos, xmat = this.data.xmat, b = this.pelvis;
    if (xpos[3 * b + 2] < (f.pelvis_z ?? 0.12)) return 'pelvis';
    if (xmat[9 * b + 8] < (f.up_dot_min ?? 0.0)) return 'tilt';
    if (this.headBody >= 0 && xpos[3 * this.headBody + 2] < (f.head_z ?? 0.3)) return 'head';
    return null;
  }
}
