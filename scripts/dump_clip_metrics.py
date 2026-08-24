"""Per-clip metrics CSV over the WHOLE corpus (train + test).

Runs the success-gated robust protocol (same core as eval_success_gated) over
every clip and writes a CSV mapping each clip -> success rate + MPKPE (global
and root-frame), tagged with its split membership.

MPKPE columns are survivor-gated (averaged only over the clip's SUCCESSFUL
passes, matching the headline metric); they are blank for clips that never
completed a pass, since MPKPE-on-survivors is undefined there. `kpe_mm`/
`pose_mm` (last-resolution, includes failure passes) are also emitted for
reference.

Usage:
    uv run python scripts/dump_clip_metrics.py <ckpt.pt|pd> \
        --cache cache/ \
        --split rgmt/data/splits/medium.json \
        --out outputs/clip_metrics.csv \
        [--n-envs 256 --action-noise 0.05 --repeats 3 --seed 0]
"""
import argparse
import csv

import torch

from rgmt.data.corpus import MotionCorpus
from rgmt.data.cache_key import file_sha256
from rgmt.data.joint_map import KEYPOINT_LINKS
from rgmt.assets.paths import ROBOT_URDF
from rgmt.eval_gated import build_eval_env, run_gated_eval, load_split


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt", help="checkpoint .pt, or 'pd' for the zero-action PD baseline")
    ap.add_argument("--cache", required=True)
    ap.add_argument("--split", default=None, help="split JSON to tag train/test membership")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-envs", type=int, default=256)
    ap.add_argument("--action-noise", type=float, default=0.05)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()

    corpus = MotionCorpus.load_cache(
        a.cache, output_device=a.device, urdf_hash=file_sha256(ROBOT_URDF),
        physics_fps=60, src_fps=30, keypoint_links=KEYPOINT_LINKS)

    # split membership (default 'train' for anything not listed as test)
    split_of = {}
    if a.split:
        sp = load_split(a.split)
        for n in sp.get("train", []):
            split_of[n] = "train"
        for n in sp.get("test", []):
            split_of[n] = "test"

    model = None
    drift_obs = drift_obs_proprio = False
    eval_kp, eval_kd, eval_action_scale, eval_filter_alpha = 100.0, 5.0, 0.5, 1.0
    eval_effort = None
    if a.ckpt != "pd":
        from rgmt.policy.networks import RGMTActorCritic, PolicyDims
        ck = torch.load(a.ckpt, map_location=a.device, weights_only=False)
        dims = PolicyDims(**ck["dims"])
        net_cfg = (ck.get("config") or {}).get("network") or {}
        model = RGMTActorCritic(dims,
            actor_hidden=tuple(net_cfg.get("actor_hidden", [512, 256])),
            critic_hidden=tuple(net_cfg.get("critic_hidden", [512, 256]))).to(a.device)
        model.load_state_dict(ck["model"])
        model.eval()
        senv = (ck.get("config") or {}).get("env") or {}
        drift_obs = bool(senv.get("drift_obs", False)) or dims.cmd_dim > 55
        drift_obs_proprio = bool(senv.get("drift_obs_proprio", False)) or dims.obs_dim > 98
        # eval plant MUST match the trained plant (per-joint gains / action_scale)
        eval_kp = senv.get("kp", 100.0)
        eval_kd = senv.get("kd", 5.0)
        eval_action_scale = float(senv.get("action_scale", 0.5))
        eval_filter_alpha = float(senv.get("action_filter_alpha", 1.0))
        eval_effort = senv.get("effort_limits", None)
        eval_dt = float(senv.get("dt", 1.0 / 60.0))
        eval_decim = int(senv.get("control_decimation", 1))
    else:
        eval_dt, eval_decim = 1.0 / 60.0, 1

    env = build_eval_env(corpus, a.device, a.n_envs,
                         drift_obs=drift_obs, drift_obs_proprio=drift_obs_proprio,
                         kp=eval_kp, kd=eval_kd, action_scale=eval_action_scale,
                         action_filter_alpha=eval_filter_alpha,
                         effort_limits=eval_effort,
                         dt=eval_dt, control_decimation=eval_decim)

    res = run_gated_eval(model, env, clip_ids=None,   # None -> every clip
                         action_noise=a.action_noise, repeats=a.repeats,
                         seed=a.seed, per_clip=True)

    rows = []
    for name in corpus.clip_names:
        rc = res["clips"].get(name, {})
        passes = rc.get("passes", 0)
        succ = rc.get("success", 0)
        _ci = corpus.clip_names.index(name)
        # clip_end is INCLUSIVE -> +1 (was off by one, review minor)
        n_frames = int(corpus.clip_end[_ci] - corpus.clip_start[_ci]) + 1
        rows.append(dict(
            clip=name,
            split=split_of.get(name, "train" if a.split else ""),
            dataset=name.split("__")[0],
            success_rate=round(succ / passes, 3) if passes else "",
            success_passes=succ,
            n_passes=passes,
            mpkpe_global_mm=rc.get("surv_kpe_mm", ""),
            mpkpe_root_mm=rc.get("surv_pose_mm", ""),
            last_pass_kpe_mm=rc.get("kpe_mm", ""),
            n_frames=n_frames,
        ))

    rows.sort(key=lambda r: (r["split"], -(r["success_rate"] if r["success_rate"] != "" else -1)))
    fields = ["clip", "split", "dataset", "success_rate", "success_passes", "n_passes",
              "mpkpe_global_mm", "mpkpe_root_mm", "last_pass_kpe_mm", "n_frames"]
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    # summary
    n = len(rows)
    surv = [r for r in rows if r["mpkpe_global_mm"] != ""]
    print(f"wrote {a.out}: {n} clips "
          f"(overall success {res['rate']*100:.1f}%, "
          f"MPKPE {res['mpkpe_mm']:.1f}mm global / {res['pose_mm']:.1f}mm root)")
    for sp in ("train", "test"):
        g = [r for r in rows if r["split"] == sp]
        if g:
            fully = sum(1 for r in g if r["success_rate"] == 1.0)
            print(f"  {sp}: {len(g)} clips, {fully} at 100% success, "
                  f"{sum(1 for r in g if r in surv)} with a survivor MPKPE")


if __name__ == "__main__":
    main()
