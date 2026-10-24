"""Bake everything the browser playground (``locomotion_webplayer/``) needs into ``locomotion_webplayer/assets/``.

    uv run python scripts/export_web.py --ckpt runs/rgmt_n24_newplant/best_test.pt

Writes:
  asimov_web.xml + meshes/   MJCF exported by Newton's SolverMuJoCo for the
                             checkpoint's plant (position actuators, torque caps,
                             contact params), with the solver options pinned in
                             ``NewtonSim`` and the visual meshes re-attached.
                             Bodies and joints carry their short MJCF names.
  policy.onnx                obs (1,98), history (1,10,98), cmd (1,21,60) -> action
                             (1,23), the pre-tanh actor mean with normalization inside.
  plant.json                 residual scale/filter, joint addresses, fall thresholds.
  motion.json/.bin           a class-stratified, mirrored subset of the locomotion
  features.bin               corpus with its motion-matching feature rows.
  golden.json                Python controller outputs for the JS port's test.
"""
from __future__ import annotations

import argparse
import json
import shutil
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import torch

from rgmt.assets.paths import ROBOT_XML
from rgmt.data.corpus import MotionCorpus, TENSOR_KEYS, clip_bookkeeping, command_from_frames
from rgmt.data.joint_map import ASIMOV_ACTUATED_JOINT_NAMES, KEYPOINT_LINKS
from rgmt.data.mirror import mirror_corpus
from rgmt.env.track_env import drift_features
from rgmt.eval_gated import plant_from_ckpt
from locomotion.command import ScriptedSource
from locomotion.config import load_config
from locomotion.controller import MotionMatchingController
from locomotion.corpus import load_locomotion_corpus
from locomotion.physics import load_policy

ROW_KEYS = ("base_pos", "base_quat", "base_lin_vel", "base_ang_vel", "joint_q", "joint_qd")
GOLDEN_SCRIPT = "w:2,q:1,wd:1,s:1,e:1"


# ---- MJCF ------------------------------------------------------------------------
def export_mjcf(plant: dict, out: Path, device: str) -> None:
    import newton
    from rgmt.env.sim import NewtonSim
    sim = NewtonSim(1, kp=plant["kp"], kd=plant["kd"], control_decimation=1, dt=plant["dt"],
                    foot_friction=0.75, keypoint_links=KEYPOINT_LINKS, device=device,
                    effort_limit=plant["effort_limits"])
    with tempfile.TemporaryDirectory() as td:
        raw = Path(td) / "newton.xml"
        newton.solvers.SolverMuJoCo(sim.model, iterations=100, ls_iterations=50, cone="pyramidal",
                                    impratio=1.0, njmax=256, nconmax=400, save_to_mjcf=str(raw))
        tree = ET.parse(raw)
    root = tree.getroot()
    orig = ET.parse(ROBOT_XML).getroot()

    # Newton prefixes every name with its path; restore the short MJCF names.
    short_bodies = [b.get("name") for b in orig.iter("body")]
    short_joints = [j.get("name") for t in ("joint", "freejoint") for j in orig.iter(t) if j.get("name")]

    def shorten(name: str, pool) -> str:
        hits = [s for s in pool if name == s or name.endswith("_" + s)]
        if not hits:
            raise KeyError(f"no short name for {name}")
        return max(hits, key=len)

    for b in root.iter("body"):
        b.set("name", shorten(b.get("name"), short_bodies))
    for j in root.iter("joint"):
        j.set("name", shorten(j.get("name"), short_joints))
    for a in root.iter("general"):
        a.set("joint", shorten(a.get("joint"), short_joints))
        a.set("name", a.get("joint"))
    for ex in root.iter("exclude"):
        ex.set("body1", shorten(ex.get("body1"), short_bodies))
        ex.set("body2", shorten(ex.get("body2"), short_bodies))

    # Solver options NewtonSim passes to SolverMuJoCo (not all are written by save_to_mjcf).
    opt = root.find("option")
    opt.set("timestep", repr(float(plant["dt"])))
    for k, v in {"iterations": "100", "ls_iterations": "50", "cone": "pyramidal", "impratio": "1",
                 "solver": "Newton"}.items():
        opt.set(k, v)
    root.find("compiler").set("meshdir", "meshes")

    # Re-attach the visual meshes (massless, collision-free) from the vendored MJCF.
    asset = root.find("asset")
    if asset is None:
        asset = ET.Element("asset")
        root.insert(list(root).index(opt) + 1, asset)
    ET.SubElement(asset, "material", name="visual_grey", rgba="0.7 0.7 0.7 1.0")
    used = {}
    for m in orig.find("asset").iter("mesh"):
        used[m.get("name")] = m.get("file")
    bodies = {b.get("name"): b for b in root.iter("body")}
    meshes = set()
    for ob in orig.iter("body"):
        for g in ob.findall("geom"):
            if g.get("class") != "visual":
                continue
            attrs = {"type": "mesh", "mesh": g.get("mesh"), "contype": "0", "conaffinity": "0",
                     "group": "2", "density": "0", "material": g.get("material", "visual_grey")}
            for k in ("name", "pos", "quat", "rgba"):
                if g.get(k) is not None:
                    attrs[k] = g.get(k)
            tb = bodies[ob.get("name")]
            # geoms must precede child bodies in MJCF
            idx = next((i for i, c in enumerate(tb) if c.tag == "body"), len(tb))
            tb.insert(idx, ET.Element("geom", attrs))
            meshes.add(g.get("mesh"))
    for name in sorted(meshes):
        ET.SubElement(asset, "mesh", name=name, file=used[name])
    for m in orig.find("asset").iter("material"):
        if m.get("name") != "visual_grey" and m.get("texture") is None:
            ET.SubElement(asset, "material", m.attrib)

    out.mkdir(parents=True, exist_ok=True)
    mdir = out / "meshes"
    mdir.mkdir(exist_ok=True)
    src = Path(ROBOT_XML).parent / orig.find("compiler").get("meshdir")
    for name in meshes:
        shutil.copy2(src / used[name], mdir / used[name])
    ET.indent(tree)
    tree.write(out / "asimov_web.xml")


def check_mjcf(path: Path) -> dict:
    import mujoco
    m = mujoco.MjModel.from_xml_path(str(path))
    if m.nu != 23:
        raise RuntimeError(f"{path}: nu={m.nu}, expected 23")
    names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, m.actuator_trnid[i, 0]) for i in range(m.nu)]
    jid = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n) for n in ASIMOV_ACTUATED_JOINT_NAMES]
    return {
        "joint_names": list(ASIMOV_ACTUATED_JOINT_NAMES),
        "qpos_adr": [int(m.jnt_qposadr[j]) for j in jid],
        "qvel_adr": [int(m.jnt_dofadr[j]) for j in jid],
        "ctrl_idx": [names.index(n) for n in ASIMOV_ACTUATED_JOINT_NAMES],
        "nq": int(m.nq), "nv": int(m.nv), "nu": int(m.nu),
    }


# ---- policy ------------------------------------------------------------------------
class _Actor(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, obs, history, cmd):
        return self.model.act_inference({"obs": obs, "history": history, "cmd_window": cmd})


def export_onnx(model, dims: dict, path: Path) -> float:
    actor = _Actor(model).eval()
    ex = (torch.randn(1, dims["obs_dim"]), torch.randn(1, dims["hist_len"], dims["obs_dim"]),
          torch.randn(1, 2 * 10 + 1, dims["cmd_dim"]))
    torch.backends.mha.set_fastpath_enabled(False)    # the fused MHA kernel has no ONNX symbolic
    with torch.no_grad():
        torch.onnx.export(actor, ex, str(path), input_names=["obs", "history", "cmd"],
                          output_names=["action"], opset_version=17, dynamo=False)
    try:
        import onnxruntime as ort
    except ImportError:
        print("onnxruntime not installed: skipping parity check (uv run --with onnxruntime ...)")
        return float("nan")
    sess = ort.InferenceSession(str(path))
    err = 0.0
    for _ in range(5):
        x = tuple(torch.randn_like(t) for t in ex)
        with torch.no_grad():
            ref = actor(*x).numpy()
        got = sess.run(None, {"obs": x[0].numpy(), "history": x[1].numpy(), "cmd": x[2].numpy()})[0]
        err = max(err, float(np.abs(ref - got).max()))
    return err


# ---- corpus ------------------------------------------------------------------------
def subset_corpus(corpus: MotionCorpus, keep: list[int]) -> MotionCorpus:
    idx = torch.cat([torch.arange(int(corpus.clip_start[c]), int(corpus.clip_end[c]) + 1) for c in keep])
    lens = corpus.clip_len[keep].clone()
    starts, ends, fcid = clip_bookkeeping(lens)
    return MotionCorpus(
        physics_fps=corpus.physics_fps, src_fps=corpus.src_fps, actuated_idx=corpus.actuated_idx,
        actuated_names=list(corpus.actuated_names), keypoint_links=list(corpus.keypoint_links),
        output_device=torch.device("cpu"), clip_start=starts, clip_len=lens, clip_end=ends,
        frame_clip_id=fcid, clip_names=[corpus.clip_names[c] for c in keep],
        **{k: getattr(corpus, k)[idx.to(corpus.storage_device)].cpu() for k in TENSOR_KEYS})


def pick_clips(corpus: MotionCorpus, meta, per_class: int, min_len: int, max_len: int,
               turners: int = 60) -> list[int]:
    by_cls: dict[str, list[int]] = {}
    for i, n in enumerate(corpus.clip_names):
        by_cls.setdefault(meta.class_of(n), []).append(i)
    keep = []
    for cls, ids in sorted(by_cls.items()):
        if cls in ("idle", "turn"):
            keep += ids                               # rare and needed: keep all
            continue
        ok = [i for i in ids if min_len <= int(corpus.clip_len[i]) <= max_len]
        rng = np.random.default_rng(0)
        keep += sorted(rng.choice(ok, size=min(per_class, len(ok)), replace=False).tolist())
    # Turning in place needs clips that turn while slow; a random draw rarely has them.
    wz = torch.stack([corpus.base_ang_vel[int(corpus.clip_start[i]):int(corpus.clip_end[i]) + 1, 2].abs().mean()
                      for i in range(corpus.n_clips)])
    v = torch.stack([corpus.base_lin_vel[int(corpus.clip_start[i]):int(corpus.clip_end[i]) + 1, :2].norm(dim=-1).mean()
                     for i in range(corpus.n_clips)])
    slow = [int(i) for i in torch.argsort(wz, descending=True) if v[i] < 0.3 and int(i) not in keep]
    keep += slow[:turners]
    return sorted(keep)


def frame_dict(f: dict) -> dict:
    return {k: [round(float(x), 7) for x in f[k].reshape(-1)] for k in ROW_KEYS}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default="locomotion_webplayer/assets")
    ap.add_argument("--per-class", type=int, default=25)
    ap.add_argument("--min-len", type=int, default=240)
    ap.add_argument("--max-len", type=int, default=900)
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--skip", nargs="*", default=[], choices=["mjcf", "onnx", "motion"])
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    model, ck = load_policy(args.ckpt, "cpu")
    plant = plant_from_ckpt(ck)
    cfg = load_config()

    if "mjcf" not in args.skip:
        export_mjcf(plant, out, args.device)
    addr = check_mjcf(out / "asimov_web.xml")
    print(f"mjcf: nq={addr['nq']} nv={addr['nv']} nu={addr['nu']}")
    (out / "plant.json").write_text(json.dumps({
        "dt": plant["dt"], "action_scale": plant["action_scale"], "alpha": plant["action_filter_alpha"],
        "kp": list(plant["kp"]), "kd": list(plant["kd"]), "caps": list(plant["effort_limits"]),
        "body_frame_ang_vel": bool(plant["body_frame_ang_vel"]), "drift_obs": bool(plant["drift_obs"]),
        **{k: addr[k] for k in ("joint_names", "qpos_adr", "qvel_adr", "ctrl_idx")},
        "fall": {"pelvis_z": 0.12, "up_dot_min": 0.0, "head_body": "neck_pitch_link", "head_z": 0.3},
        "max_drift": cfg.physics.max_drift, "hist_len": ck["dims"]["hist_len"], "cmd_len": ck["dims"]["cmd_len"],
        "obs_dim": ck["dims"]["obs_dim"], "cmd_dim": ck["dims"]["cmd_dim"], "iteration": ck.get("iteration"),
    }, indent=1))

    if "onnx" not in args.skip:
        err = export_onnx(model, ck["dims"], out / "policy.onnx")
        print(f"onnx: max |torch - ort| = {err:.2e}")

    if "motion" in args.skip:
        return 0
    cfg.corpus.mirror = False
    full, meta = load_locomotion_corpus(cfg, "cpu")
    keep = pick_clips(full, meta, args.per_class, args.min_len, args.max_len)
    corpus = mirror_corpus(subset_corpus(full, keep))
    ctl = MotionMatchingController(corpus, meta, cfg, device="cpu")
    db = ctl.db
    rows = torch.cat([getattr(corpus, k).reshape(corpus.n_frames, -1) for k in ROW_KEYS], 1).float()
    assert rows.shape[1] == 59
    rows.numpy().astype("<f4").tofile(out / "motion.bin")
    db.raw.numpy().astype("<f4").tofile(out / "features.bin")
    lay, a = {}, 0
    for k in ROW_KEYS:
        n = getattr(corpus, k).reshape(corpus.n_frames, -1).shape[1]
        lay[k] = [a, a + n]; a += n
    from omegaconf import OmegaConf
    cfg_dict = OmegaConf.to_container(OmegaConf.structured(cfg))
    cfg_dict.pop("corpus", None)
    (out / "motion.json").write_text(json.dumps({
        "n_frames": corpus.n_frames, "row": 59, "layout": lay,
        "clip_start": corpus.clip_start.tolist(), "clip_end": corpus.clip_end.tolist(),
        "clip_names": corpus.clip_names, "clip_class": [meta.class_of(n) for n in corpus.clip_names],
        "speeds": ctl.speeds, "config": cfg_dict,
        "feat": {"dim": db.dim, "pose_dim": db.pose_dim, "mean": db.mean.tolist(),
                 "scale": db.scale.tolist(), "weight": db.weight.tolist()},
    }))
    print(f"motion: {len(keep)} clips x2 mirrored, {corpus.n_frames} frames, "
          f"{(out / 'motion.bin').stat().st_size / 1e6:.1f} MB + {(out / 'features.bin').stat().st_size / 1e6:.1f} MB")

    # golden fixture for locomotion_webplayer/test/controller.test.mjs
    ctl.reset()
    src = ScriptedSource(GOLDEN_SCRIPT, fps=cfg.fps)
    cmds, frames, previews, past = [], [], {}, []
    t = 0
    while (c := src.next()) is not None:
        f = ctl.step(c)
        cmds.append([float(c.move[0]), float(c.move[1]), float(c.yaw)])
        frames.append(frame_dict(f))
        past.append(f)
        if t % 30 == 0:
            p = ctl.preview(10, blended=True)
            previews[str(t)] = [frame_dict({k: p[k][i] for k in ROW_KEYS}) for i in range(10)]
        t += 1
    win_rows = past[-21:]
    win = {k: torch.cat([r[k] for r in win_rows], 0) for k in ROW_KEYS}
    robot_pos = win["base_pos"][10] + torch.tensor([0.1, -0.05, -0.02])
    robot_quat = torch.tensor([0.0, 0.0, 0.2588190, 0.9659258])   # 30 deg yaw, xyzw
    drift = drift_features(win["base_pos"][None], win["base_quat"][None], robot_pos[None], robot_quat[None])
    (out / "golden.json").write_text(json.dumps({
        "script": GOLDEN_SCRIPT, "cmds": cmds, "frames": frames, "previews": previews,
        "window": [frame_dict({k: win[k][i] for k in ROW_KEYS}) for i in range(21)],
        "robot_pos": robot_pos.tolist(), "robot_quat": robot_quat.tolist(),
        "cmd_feats": command_from_frames(win).reshape(-1).tolist(), "drift": drift.reshape(-1).tolist(),
    }))
    print(f"golden: {t} ticks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
