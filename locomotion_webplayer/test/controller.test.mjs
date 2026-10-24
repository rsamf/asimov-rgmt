// Golden test of the JS controller port against the Python MotionMatchingController.
// Run: node locomotion_webplayer/test/controller.test.mjs   (needs locomotion_webplayer/assets/ from the bake)
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import { makeMotion, fromKeys, Controller } from "../controller.js";
import { commandFromFrames, driftFeatures } from "../features.js";

const here = dirname(fileURLToPath(import.meta.url));
const assets = process.argv[2] || join(here, "..", "assets");
const f32 = (p) => { const b = readFileSync(p); return new Float32Array(b.buffer, b.byteOffset, b.byteLength / 4); };
const json = JSON.parse(readFileSync(join(assets, "motion.json"), "utf8"));
const golden = JSON.parse(readFileSync(join(assets, "golden.json"), "utf8"));

const t0 = performance.now();
const motion = makeMotion(json, f32(join(assets, "motion.bin")), f32(join(assets, "features.bin")));
console.log(`makeMotion: ${motion.N} frames, ${motion.validIdx.length} valid, ${(performance.now() - t0).toFixed(0)} ms`);

const KEYS = ["base_pos", "base_quat", "base_lin_vel", "base_ang_vel", "joint_q", "joint_qd"];
const TOL = { base_pos: 1e-3, base_quat: 1e-4, base_lin_vel: 1e-3, base_ang_vel: 1e-3, joint_q: 1e-4, joint_qd: 1e-3 };
let failed = false;
const maxErr = {};
const firstBad = {};
const upd = (field, a, b, tick) => {
  let e = 0;
  for (let i = 0; i < b.length; i++) e = Math.max(e, Math.abs(a[i] - b[i]));
  if (Number.isNaN(e) || maxErr[field] === undefined || e > maxErr[field]) { if (!Number.isNaN(maxErr[field])) maxErr[field] = e; }
  if (!(e <= TOL[field.split(":").pop()]) && firstBad[field] === undefined) firstBad[field] = tick;
  return e;
};
const cmpFrame = (prefix, a, b, tick) => { for (const k of KEYS) upd(`${prefix}:${k}`, a[k], b[k].flat(), tick); };

// 1. script -> commands through fromKeys
const fps = json.config.fps;
const cmdsJs = [];
for (const seg of golden.script.split(",")) {
  const [keys, secs] = seg.trim().split(":");
  const c = fromKeys(new Set(keys.trim().toLowerCase()));
  for (let i = 0; i < Math.round(Number(secs) * fps); i++) cmdsJs.push([c.move[0], c.move[1], c.yaw]);
}
let cmdErr = cmdsJs.length === golden.cmds.length ? 0 : Infinity;
for (let i = 0; i < Math.min(cmdsJs.length, golden.cmds.length); i++)
  for (let j = 0; j < 3; j++) cmdErr = Math.max(cmdErr, Math.abs(cmdsJs[i][j] - golden.cmds[i][j]));
console.log(`fromKeys vs golden cmds (${cmdsJs.length} vs ${golden.cmds.length} ticks): max err ${cmdErr}`);
if (!(cmdErr <= 1e-12)) failed = true;

// 2. rollout
const ctl = new Controller(motion);
ctl.reset();
let tStep = 0;
for (let t = 0; t < golden.cmds.length; t++) {
  const [mx, my, yaw] = golden.cmds[t];
  const s0 = performance.now();
  const fr = ctl.step({ move: [mx, my], yaw });
  tStep += performance.now() - s0;
  cmpFrame("step", fr, golden.frames[t], t);
  const pv = golden.previews?.[String(t)];
  if (pv) { const p = ctl.preview(pv.length); for (let i = 0; i < pv.length; i++) cmpFrame("preview", p[i], pv[i], t); }
}
console.log(`rollout: ${golden.cmds.length} ticks, ${(tStep / golden.cmds.length).toFixed(2)} ms/step avg`);

// 3. command / drift features on the golden window
if (golden.window) {
  const cf = commandFromFrames(golden.window);
  maxErr["cmd_feats"] = 0; maxErr["drift"] = 0;
  for (let i = 0; i < cf.length; i++) maxErr["cmd_feats"] = Math.max(maxErr["cmd_feats"], Math.abs(cf[i] - golden.cmd_feats[i]));
  const df = driftFeatures(golden.window, golden.robot_pos, golden.robot_quat);
  for (let i = 0; i < df.length; i++) maxErr["drift"] = Math.max(maxErr["drift"], Math.abs(df[i] - golden.drift[i]));
  if (cf.length !== golden.cmd_feats.length || df.length !== golden.drift.length) { console.log("feature length mismatch"); failed = true; }
  TOL.cmd_feats = 1e-4; TOL.drift = 1e-4;
}

for (const [k, e] of Object.entries(maxErr)) {
  const tol = TOL[k.split(":").pop()];
  const ok = e <= tol;
  if (!ok) failed = true;
  console.log(`${ok ? "PASS" : "FAIL"} ${k.padEnd(24)} max abs err ${e.toExponential(3)} (tol ${tol})` +
    (firstBad[k] !== undefined ? `  first over tol at tick ${firstBad[k]}` : ""));
}
console.log(failed ? "FAILED" : "ALL PASS");
process.exit(failed ? 1 : 0);
