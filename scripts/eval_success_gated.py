"""Success-gated, per-clip eval — the paper-comparable protocol.

Thin CLI over rgmt.eval_gated (the same core the training loop runs
in-process since 2026-07-15). Protocol and output format unchanged.

Rolls EVERY clip from its start to its end (no RSI sampling): success = the
episode reaches the clip end without any failure termination. Reports the
success rate and tracking metrics computed ONLY over steps of successful
episodes (paper-style MPJPE-on-survivors), so precision and survival stop
being conflated.

Usage: python scripts/eval_success_gated.py <ckpt.pt|pd> [n_envs] [--limit N] [--gains P --effort none|urdf[:scale] --torque]
           [--cache DIR] [--dump-clips out.json]
           [--action-noise SIGMA] [--repeats K]
           [--split split.json --role test|train]

--dump-clips writes per-clip outcomes {name: {success, steps, kpe_mm}}.
With --repeats K, "success" becomes the count of passes (0..K) completed.
--action-noise SIGMA adds N(0, SIGMA) to pre-tanh actions each step:
single deterministic passes proved fragile — noise + repeats measures the
policy's success NEIGHBORHOOD, which is the quantity that transfers.
--split/--role restricts eval to one side of a train/test split
(e.g. rgmt/data/splits/medium.json); mutually exclusive with --limit.
"""
import json
import sys

import torch

from rgmt.data.corpus import MotionCorpus
from rgmt.data.cache_key import file_sha256
from rgmt.data.joint_map import KEYPOINT_LINKS
from rgmt.assets.paths import ROBOT_URDF
from rgmt.eval_gated import build_eval_env, run_gated_eval, load_split, split_clip_ids, plant_from_ckpt

CACHE = "cache/"
dev = "cuda:0"

ckpt_path = sys.argv[1]
n_envs = int(sys.argv[2]) if len(sys.argv) > 2 and not sys.argv[2].startswith("--") else 256
limit = None
if "--limit" in sys.argv:
    limit = int(sys.argv[sys.argv.index("--limit") + 1])
if "--cache" in sys.argv:
    CACHE = sys.argv[sys.argv.index("--cache") + 1]
dump_path = None
if "--dump-clips" in sys.argv:
    dump_path = sys.argv[sys.argv.index("--dump-clips") + 1]
action_noise = 0.0
if "--action-noise" in sys.argv:
    action_noise = float(sys.argv[sys.argv.index("--action-noise") + 1])
repeats = 1
if "--repeats" in sys.argv:
    repeats = int(sys.argv[sys.argv.index("--repeats") + 1])
split_path = None
if "--split" in sys.argv:
    split_path = sys.argv[sys.argv.index("--split") + 1]
role = sys.argv[sys.argv.index("--role") + 1] if "--role" in sys.argv else "test"
if split_path is not None and limit is not None:
    sys.exit("--split and --limit are mutually exclusive")

corpus = MotionCorpus.load_cache(
    CACHE, output_device=dev, urdf_hash=file_sha256(ROBOT_URDF),
    physics_fps=60, src_fps=30, keypoint_links=KEYPOINT_LINKS)

drift_obs = drift_obs_proprio = False
model = None
if ckpt_path != "pd":
    from rgmt.policy.networks import RGMTActorCritic, PolicyDims
    ck = torch.load(ckpt_path, map_location=dev, weights_only=False)
    dims = PolicyDims(**ck["dims"])
    net_cfg = (ck.get("config") or {}).get("network") or {}
    model = RGMTActorCritic(dims,
        actor_hidden=tuple(net_cfg.get("actor_hidden", [512, 256])),
        critic_hidden=tuple(net_cfg.get("critic_hidden", [512, 256]))).to(dev)
    model.load_state_dict(ck["model"])
    model.eval()
    # eval plant MUST match the trained plant: everything comes from the checkpoint.
    plant = plant_from_ckpt(ck)
else:
    # Pure-PD floor. --gains <profile> and --effort none|urdf choose the plant
    # (defaults: the plant-v1 recipe's leg-weighted gains, uncapped).
    from rgmt.env.gains import gains_profile, urdf_effort_limits
    _prof = sys.argv[sys.argv.index("--gains") + 1] if "--gains" in sys.argv else "leg_weighted"
    _eff = sys.argv[sys.argv.index("--effort") + 1] if "--effort" in sys.argv else "none"
    _kp, _kd = gains_profile(_prof)
    # --effort urdf:<scale> scales every cap (e.g. 0.8 = DR's lower edge).
    _eff_scale = float(_eff.split(":", 1)[1]) if _eff.startswith("urdf:") else 1.0
    plant = dict(kp=_kp, kd=_kd, action_scale=0.3, action_filter_alpha=1.0,
                 effort_limits=([c * _eff_scale for c in urdf_effort_limits()]
                                if _eff.startswith("urdf") else None))
    print(f"PD plant: gains={_prof} effort={_eff}")
torque_report = "--torque" in sys.argv

env = build_eval_env(corpus, dev, n_envs, **plant)

if split_path is not None:
    clip_ids = split_clip_ids(corpus, load_split(split_path), role)
    tag = f"{len(clip_ids)} clips, split={role}"
else:
    n_clips = len(corpus.clip_names) if limit is None else min(limit, len(corpus.clip_names))
    clip_ids = range(n_clips)
    tag = f"{n_clips} clips"

r = run_gated_eval(model, env, clip_ids, action_noise=action_noise,
                   repeats=repeats, seed=0, per_clip=dump_path is not None, report_torque=torque_report)

print(f"=== success-gated eval: {ckpt_path} ({tag}) ===")
print(f"  success rate:            {r['success']}/{r['total']} = {r['rate']*100:.1f} %")
if repeats > 1:
    m = sum(r["per_pass"]) / repeats
    sd = (sum((x - m) ** 2 for x in r["per_pass"]) / repeats) ** 0.5
    n = r["n_clips"]
    rates = "/".join(f"{x/n*100:.1f}" for x in r["per_pass"])
    print(f"  per-pass (noise {action_noise}): {r['per_pass']} = {rates} %  "
          f"mean {m/n*100:.1f} % sd {sd/n*100:.1f} %")
print(f"  MPKPE on survivors:      {r['mpkpe_mm']:6.1f} mm")
print(f"  pose (root frame), surv: {r['pose_mm']:6.1f} mm")
print(f"  jitter (cmd), surv:      {r['jitter_mrad']:6.1f} mrad/step")
if r.get("torque"):
    rows = sorted(r["torque"].items(), key=lambda kv: -kv[1]["sat_frac"])
    print("  torque demand vs cap (counted steps): joint  sat_frac  mean|tau|/cap")
    for name, v in rows[:8]:
        print(f"    {name.replace('_joint',''):22s} {v['sat_frac']*100:6.2f} %   {v['mean_ratio']:.3f}")
    worst = max(v["sat_frac"] for v in r["torque"].values())
    print(f"  worst-joint saturation: {worst*100:.2f} %")

# Greedy (zero-noise) reference, the deployment-honest number. Pre-tanh
# eval noise can act as beneficial dither (measured ~+3 pts on a policy
# trained without domain randomization), so the robust rate can flatter the
# deterministic policy. Reports should quote BOTH. Skip with --no-greedy.
if action_noise > 0.0 and "--no-greedy" not in sys.argv:
    rg = run_gated_eval(model, env, clip_ids, action_noise=0.0,
                        repeats=1, seed=0, per_clip=False)
    print(f"  greedy (deploy) success: {rg['success']}/{rg['total']} = "
          f"{rg['rate']*100:.1f} %   mpkpe {rg['mpkpe_mm']:.1f} mm")
if dump_path is not None:
    with open(dump_path, "w") as f:
        json.dump(r["clips"], f, indent=0, sort_keys=True)
    print(f"  per-clip outcomes -> {dump_path} ({len(r['clips'])} clips)")
