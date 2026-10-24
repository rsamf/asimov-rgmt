// Browser port of locomotion/physics.py PhysicsDriver: motion-matching
// controller -> 21-frame command window -> ONNX policy residual -> PD in MuJoCo.
import { Sim, yawOfQuat } from './sim.js';
import { Renderer } from './render.js';
import { Policy } from './policy.js';

const params = new URLSearchParams(location.search);
const BASE = params.get('assets') || 'assets/';
const MOTION_BASE = params.get('motion') || 'assets/';
const $ = (id) => document.getElementById(id);
const showError = (m) => (window.__showError ? window.__showError(m) : console.error(m));

function progress(msg, frac) {
  $('msg').textContent = msg;
  if (frac != null) $('bar').firstElementChild.style.width = `${Math.round(100 * frac)}%`;
}

// ---- controller (falls back to a standing stub if controller.js is missing) ----
function stubController(nAct, height = 0.78) {
  let pos = [0, 0], yaw = 0;
  const frame = () => ({
    base_pos: [pos[0], pos[1], height], base_quat: [0, 0, Math.sin(yaw / 2), Math.cos(yaw / 2)],
    base_lin_vel: [0, 0, 0], base_ang_vel: [0, 0, 0],
    joint_q: new Array(nAct).fill(0), joint_qd: new Array(nAct).fill(0),
  });
  return {
    ctl: {
      reset(xy = [0, 0], y = 0) { pos = [...xy]; yaw = y; },
      step() { return frame(); },
      preview(n) { return Array.from({ length: n }, frame); },
      translate(d) { pos[0] += d[0]; pos[1] += d[1]; },
      get worldYaw() { return yaw; },
    },
    fromKeys: () => ({ move: [0, 0], yaw: 0 }),
    commandFromFrames: (fr) => new Float32Array(fr.length * 55),
    driftFeatures: (fr) => new Float32Array(fr.length * 5),
    stub: true,
  };
}

async function loadController(nAct) {
  try {
    const C = await import('./controller.js');
    const F = await import('./features.js');
    progress('Loading motion database…');
    const motion = await C.loadMotion(MOTION_BASE);
    return { ctl: new C.Controller(motion), fromKeys: C.fromKeys,
             commandFromFrames: F.commandFromFrames, driftFeatures: F.driftFeatures, stub: false };
  } catch (e) {
    console.warn('[main] controller unavailable, using standing stub:', e);
    return stubController(nAct);
  }
}

const cloneFrame = (f) => ({ ...f, base_pos: [f.base_pos[0], f.base_pos[1], f.base_pos[2]] });

class Driver {
  constructor(sim, policy, C, plant) {
    this.sim = sim; this.policy = policy; this.C = C; this.plant = plant;
    this.L = plant.hist_len ?? 10;
    this.O = plant.obs_dim ?? 98; this.A = plant.joint_names.length;
    this.W = plant.cmd_len ?? 21; this.D = plant.cmd_dim ?? 60;
    this.prevAction = new Float32Array(this.A);
    this.filt = new Float64Array(this.A);
    this.obs = new Float32Array(this.O);
    this.history = new Float32Array(this.L * this.O);   // hist_len copies, oldest first
    this.cmd = new Float32Array(this.W * this.D);
    this.zeros = new Float32Array(this.A);
    this.past = []; this.cur = null;
    this.falls = 0;
    this.physics = true;          // false: the controller drives the robot kinematically
    this.target = new Float64Array(this.A);
  }

  reset(xy = [0, 0], yaw = 0) {
    const { ctl } = this.C;
    ctl.reset(xy, yaw);
    this.cur = cloneFrame(ctl.step({ move: [0, 0], yaw: 0 }));
    this.sim.teleport(this.cur);
    this.past = Array.from({ length: this.L }, () => cloneFrame(this.cur));
    this._warmStart();
  }

  // policy state for a robot standing on `cur` (after a teleport)
  _warmStart() {
    this.prevAction.fill(0); this.filt.fill(0);
    this.sim.buildObs(this.prevAction, this.obs);
    for (let i = 0; i < this.L; i++) this.history.set(this.obs, i * this.O);
  }

  // Switch modes without resetting the controller: off -> the robot snaps onto the
  // reference (within max_drift); on -> physics starts from the reference pose and
  // velocities mid-stride, like reference-state init in training.
  setPhysics(on) {
    if (on === this.physics) return;
    this.physics = on;
    if (on) { this.sim.teleport(this.cur); this._warmStart(); }
  }

  tickKinematic(command) {
    const t0 = performance.now();
    this.past.push(this.cur); if (this.past.length > this.L) this.past.shift();
    this.cur = cloneFrame(this.C.ctl.step(command));
    this.sim.teleport(this.cur);
    return { fell: false, simMs: performance.now() - t0, policyMs: 0 };
  }

  buildCmd() {
    const win = [...this.past, this.cur, ...this.C.ctl.preview(this.L)];
    const g = this.C.commandFromFrames(win);                                   // 21*55
    const nCmd = this.D - 5;
    const drift = this.D > nCmd ? this.C.driftFeatures(win, this.sim.rootPos(), this.sim.rootQuat()) : null;
    for (let t = 0; t < this.W; t++) {
      this.cmd.set(g.subarray(t * nCmd, (t + 1) * nCmd), t * this.D);
      if (drift) this.cmd.set(drift.subarray(t * 5, (t + 1) * 5), t * this.D + nCmd);
    }
  }

  // one 60 Hz tick; returns {fell, simMs, policyMs}
  async tick(command) {
    if (!this.physics) return this.tickKinematic(command);
    const P = this.plant, A = this.A;
    let action = this.zeros, policyMs = 0;
    if (this.policy) {
      const t0 = performance.now();
      this.sim.buildObs(this.prevAction, this.obs);
      this.buildCmd();
      action = await this.policy.run(this.obs, this.history, this.cmd);
      policyMs = performance.now() - t0;
    }
    const t1 = performance.now();
    const alpha = P.alpha ?? 1.0, scale = P.action_scale;
    for (let i = 0; i < A; i++) {
      const ta = Math.tanh(action[i]);
      let res = ta * scale;
      if (alpha < 1.0) { this.filt[i] = alpha * res + (1 - alpha) * this.filt[i]; res = this.filt[i]; }
      this.target[i] = this.cur.joint_q[i] + res;
      this.prevAction[i] = ta;
    }
    this.sim.step(this.target);

    this.past.push(this.cur); if (this.past.length > this.L) this.past.shift();
    this.cur = cloneFrame(this.C.ctl.step(command));
    this.leash();
    this.sim.buildObs(this.prevAction, this.obs);
    this.history.copyWithin(0, this.O);
    this.history.set(this.obs, (this.L - 1) * this.O);
    const simMs = performance.now() - t1;

    const why = this.sim.fallen();
    if (why) {
      this.falls++;
      const p = this.sim.rootPos();
      console.log(`[main] fall (${why}) #${this.falls}`);
      this.reset([p[0], p[1]], this.C.ctl.worldYaw ?? 0);
      return { fell: true, simMs, policyMs };
    }
    return { fell: false, simMs, policyMs };
  }

  leash() {
    const cap = this.plant.max_drift;
    if (cap == null) return;
    const r = this.sim.rootPos();
    const dx = this.cur.base_pos[0] - r[0], dy = this.cur.base_pos[1] - r[1];
    const n = Math.hypot(dx, dy);
    if (n <= cap) return;
    const sx = r[0] + dx * cap / n - this.cur.base_pos[0];
    const sy = r[1] + dy * cap / n - this.cur.base_pos[1];
    this.C.ctl.translate([sx, sy]);
    for (const f of [this.cur, ...this.past]) { f.base_pos[0] += sx; f.base_pos[1] += sy; }
  }

  get drift() {
    const r = this.sim.rootPos();
    return Math.hypot(this.cur.base_pos[0] - r[0], this.cur.base_pos[1] - r[1]);
  }
}

async function main() {
  progress('Loading plant…', 0);
  const plant = await (await fetch(BASE + 'plant.json')).json();
  const renderer = new Renderer($('view'));
  const sim = await Sim.load(BASE, plant, params.get('xml') || 'asimov_web.xml', progress);
  renderer.buildModel(sim.model);

  progress('Loading policy…', 0.98);
  let policy = null;
  try { policy = await Policy.load(BASE + 'policy.onnx', plant); }
  catch (e) { console.warn('[main] policy unavailable:', e); showError('Policy not loaded (pure PD only): ' + (e.message || e)); }

  const C = await loadController(plant.joint_names.length);
  const drv = new Driver(sim, policy, C, plant);
  drv.reset([0, 0], 0);
  window.__app = { drv, sim, renderer, plant, C };

  // ---- input ----
  const keys = new Set();
  const updateBadge = () => {
    const b = $('badge');
    b.className = drv.physics ? 'on' : 'kin';
    b.textContent = !drv.physics ? 'PHYSICS OFF · controller' : policy ? 'PHYSICS ON · WBC' : 'PHYSICS (PD only)';
  };
  // mode switches are applied between ticks, never while one is awaiting the policy
  const req = { reset: false, physics: null };
  const togglePhysics = () => { req.physics = !(req.physics ?? drv.physics); };
  $('badge').addEventListener('click', (e) => { togglePhysics(); e.currentTarget.blur(); });
  window.addEventListener('keydown', (e) => {
    const k = e.key.toLowerCase();
    if (k === ' ' || e.code === 'Space') { e.preventDefault(); if (!e.repeat) req.reset = true; return; }
    if (k === 'p') { if (!e.repeat) togglePhysics(); return; }
    if (k === 'v') { if (!e.repeat) renderer.showTraj = !renderer.showTraj; return; }
    if ('wasdqe'.includes(k) && k.length === 1) { keys.add(k); e.preventDefault(); }
  });
  window.addEventListener('keyup', (e) => keys.delete(e.key.toLowerCase()));
  window.addEventListener('blur', () => keys.clear());
  updateBadge();

  $('loading').style.display = 'none';

  // ---- fixed-step loop ----
  const DT = plant.dt || 1 / 60, MAX_CATCHUP = 3;
  let acc = 0, last = performance.now(), busy = false;
  let ticks = 0, statT = performance.now(), simMsAvg = 0, polMsAvg = 0, tickHz = 0;
  async function frame(now) {
    requestAnimationFrame(frame);
    acc += Math.min(0.25, (now - last) / 1000); last = now;
    if (busy) return;
    busy = true;
    try {
      if (req.physics !== null) { drv.setPhysics(req.physics); req.physics = null; updateBadge(); }
      if (req.reset) {
        req.reset = false; drv.reset([0, 0], 0); renderer.resetCamera(); acc = 0;
      }
      let n = 0;
      while (acc >= DT && n < MAX_CATCHUP) {
        const cmd = C.fromKeys(new Set(keys));
        const r = await drv.tick(cmd);
        simMsAvg += 0.05 * (r.simMs - simMsAvg); polMsAvg += 0.05 * (r.policyMs - polMsAvg);
        acc -= DT; n++; ticks++;
      }
      if (n === MAX_CATCHUP) acc = Math.min(acc, DT);
      const q = sim.rootQuat();
      renderer.setTrajectory(C.ctl.trajectory ? C.ctl.trajectory() : null);
      renderer.update(sim.data, sim.rootPos(), yawOfQuat(q), Math.max(DT, n * DT));
      if (now - statT > 500) {
        tickHz = ticks * 1000 / (now - statT); ticks = 0; statT = now;
        $('stats').innerHTML = drv.physics
          ? `sim ${tickHz.toFixed(0)} Hz<br>step ${simMsAvg.toFixed(2)} ms · policy ${polMsAvg.toFixed(2)} ms<br>` +
            `falls ${drv.falls} · drift ${drv.drift.toFixed(2)} m`
          : `controller ${tickHz.toFixed(0)} Hz · ${simMsAvg.toFixed(2)} ms<br>` +
            `${C.ctl.clipClass ?? ''}<br><span class="clip">${C.ctl.clipName ?? ''}</span>`;
        if (C.stub) $('stats').innerHTML += '<br>controller: stub';
      }
    } catch (e) {
      showError('Loop: ' + (e.stack || e)); throw e;
    } finally { busy = false; }
  }
  requestAnimationFrame(frame);
}

main().catch((e) => { showError('Startup failed: ' + (e.stack || e)); progress('Startup failed — see error panel'); });
