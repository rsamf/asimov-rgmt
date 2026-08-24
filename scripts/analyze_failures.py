"""Failure-clip autopsy for a robust-eval per-clip dump.

Buckets clips by pass-success count (0..K), clusters failures by source
dataset, clip length, survival depth, and reference dynamics (speed), and
prints representative hard-fail clips for rendering.

Usage: python scripts/analyze_failures.py <clips.json> [--repeats K] [--cache DIR]
"""
import json
import sys
from collections import defaultdict

import torch

from rgmt.data.corpus import MotionCorpus
from rgmt.data.cache_key import file_sha256
from rgmt.data.joint_map import KEYPOINT_LINKS
from rgmt.assets.paths import ROBOT_URDF

CACHE = "cache/"
path = sys.argv[1]
repeats = int(sys.argv[sys.argv.index("--repeats") + 1]) if "--repeats" in sys.argv else 3
if "--cache" in sys.argv:
    CACHE = sys.argv[sys.argv.index("--cache") + 1]

d = json.load(open(path))
corpus = MotionCorpus.load_cache(
    CACHE, output_device="cpu", urdf_hash=file_sha256(ROBOT_URDF),
    physics_fps=60, src_fps=30, keypoint_links=KEYPOINT_LINKS)
name_to_id = {n: i for i, n in enumerate(corpus.clip_names)}

def clip_stats(name):
    c = name_to_id[name]
    s, e = int(corpus.clip_start[c]), int(corpus.clip_end[c])
    v = corpus.base_lin_vel[s:e]
    speed = v[:, :2].norm(dim=-1)
    z = corpus.base_pos[s:e, 2]
    return dict(length=e - s, mean_speed=float(speed.mean()), max_speed=float(speed.max()),
                z_range=float(z.max() - z.min()))

hard = {n: r for n, r in d.items() if r["success"] == 0}
marginal = {n: r for n, r in d.items() if 0 < r["success"] < repeats}
solid = {n: r for n, r in d.items() if r["success"] == repeats}
print(f"=== {path} (K={repeats}) ===")
print(f"solid {len(solid)}  marginal {len(marginal)}  HARD-FAIL {len(hard)}  / {len(d)}")

def dataset(n):
    return n.split("__")[0]

print("\n--- hard-fail share by dataset ---")
tot = defaultdict(int); bad = defaultdict(int)
for n in d: tot[dataset(n)] += 1
for n in hard: bad[dataset(n)] += 1
for k in sorted(tot, key=lambda k: -bad[k] / max(tot[k], 1)):
    print(f"  {k:12s} {bad[k]:4d}/{tot[k]:4d} = {bad[k]/max(tot[k],1)*100:5.1f} %")

print("\n--- dynamics: hard-fail vs solid (means) ---")
for label, group in (("hard", hard), ("solid", solid)):
    st = [clip_stats(n) for n in group]
    if not st: continue
    m = {k: sum(s[k] for s in st) / len(st) for k in st[0]}
    print(f"  {label:8s} len {m['length']:6.0f}  mean_speed {m['mean_speed']:.2f} m/s  "
          f"max_speed {m['max_speed']:.2f}  z_range {m['z_range']:.2f} m")

print("\n--- survival depth on hard fails ---")
steps = sorted(r["steps"] for r in hard.values())
if steps:
    q = lambda p: steps[int(p * (len(steps) - 1))]
    print(f"  steps survived: p10 {q(.1)}  median {q(.5)}  p90 {q(.9)}  (60 steps = 1 s)")
    frac_early = sum(1 for s in steps if s < 120) / len(steps)
    print(f"  fraction dying < 2 s: {frac_early*100:.0f} %")

print("\n--- representative hard fails (long + fast first) ---")
ranked = sorted(hard, key=lambda n: -(clip_stats(n)["mean_speed"] * clip_stats(n)["length"]))
for n in ranked[:12]:
    s = clip_stats(n)
    print(f"  {n}  (len {s['length']}, mean {s['mean_speed']:.2f} m/s, "
          f"z_range {s['z_range']:.2f}, died @{hard[n]['steps']})")
