// Policy command features: ports of rgmt/data/corpus.py:command_from_frames (55 per frame)
// and rgmt/env/track_env.py:drift_features (5 per frame). Quaternions are xyzw.

function quatToMatrix(q) {
  const n = Math.max(Math.hypot(q[0], q[1], q[2], q[3]), 1e-9);
  const x = q[0] / n, y = q[1] / n, z = q[2] / n, w = q[3] / n;
  const xx = x * x, yy = y * y, zz = z * z, xy = x * y, xz = x * z, yz = y * z, wx = w * x, wy = w * y, wz = w * z;
  return [
    1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy),
    2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx),
    2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy),
  ];
}

// R^T v (world -> body frame); R is row-major 3x3
function rotInv(R, v, out, o) {
  for (let i = 0; i < 3; i++) out[o + i] = R[i] * v[0] + R[3 + i] * v[1] + R[6 + i] * v[2];
}

function yawFromQuat(q) {
  const [x, y, z, w] = q;
  return Math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z));
}

export function commandFromFrames(frames) {
  const S = frames.length, out = new Float32Array(S * 55);
  const tmp = new Float64Array(9);
  const G = [0, 0, -1];
  for (let s = 0; s < S; s++) {
    const f = frames[s], o = s * 55;
    const R = quatToMatrix(f.base_quat);
    rotInv(R, f.base_lin_vel, tmp, 0);
    rotInv(R, f.base_ang_vel, tmp, 3);
    rotInv(R, G, tmp, 6);
    for (let i = 0; i < 9; i++) out[o + i] = tmp[i];
    const q = f.joint_q, n = q.length;
    for (let i = 0; i < n; i++) { out[o + 9 + i] = Math.cos(q[i]); out[o + 9 + n + i] = Math.sin(q[i]); }
  }
  return out;
}

export function driftFeatures(frames, robotPos, robotQuat) {
  const S = frames.length, out = new Float32Array(S * 5);
  const yr = yawFromQuat(robotQuat), c = Math.cos(yr), s = Math.sin(yr);
  for (let k = 0; k < S; k++) {
    const f = frames[k], o = k * 5;
    const dx = f.base_pos[0] - robotPos[0], dy = f.base_pos[1] - robotPos[1], dz = f.base_pos[2] - robotPos[2];
    const dyaw = yawFromQuat(f.base_quat) - yr;
    out[o] = c * dx + s * dy;
    out[o + 1] = -s * dx + c * dy;
    out[o + 2] = dz;
    out[o + 3] = Math.cos(dyaw);
    out[o + 4] = Math.sin(dyaw);
  }
  return out;
}
