"""Side-by-side render: reference clip (left) vs its left-right mirror (right).

Pure kinematic playback of the corpus data through MuJoCo's offscreen EGL
renderer, no physics: what you see is exactly what ``mirror_augment`` feeds
the trainer. Both panes share the camera and are recentred on their own first
root position so a lateral drift reads as motion, not as an offset.

Usage:
    uv run --with imageio --with imageio-ffmpeg python scripts/render_mirror.py \
        --cache <corpus dir> --out-dir outputs/mirror "<clip name>" ["<clip name>" ...]
    uv run --with imageio --with imageio-ffmpeg python scripts/render_mirror.py \
        --cache <corpus dir> --grep "turn left" --limit 2

Writes ``<out-dir>/<sanitised clip name>__mirror.mp4`` per clip at 60 fps.
"""
import argparse
import os
import re
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np
import torch
import imageio.v2 as iio

from rgmt.assets.paths import ROBOT_URDF, ROBOT_XML
from rgmt.data.cache_key import file_sha256
from rgmt.data.corpus import MotionCorpus
from rgmt.data.joint_map import KEYPOINT_LINKS
from rgmt.data.mirror import MIRROR_SUFFIX, mirror_corpus


def xyzw_to_wxyz(q):
    return np.concatenate([q[..., 3:4], q[..., 0:3]], axis=-1)


def clip_qpos(corpus, name, n_hinges):
    """(n, nq) MuJoCo qpos rows for one clip: root pos, wxyz quat, all hinges."""
    ci = corpus.clip_names.index(name)
    s, e = int(corpus.clip_start[ci]), int(corpus.clip_end[ci])
    ref = corpus.at(torch.arange(s, e + 1))
    n = e - s + 1
    hinges = np.zeros((n, n_hinges), dtype=np.float32)
    hinges[:, corpus.actuated_idx.cpu().numpy()] = ref["joint_q"].cpu().numpy()
    q = np.concatenate([ref["base_pos"].cpu().numpy(),
                        xyzw_to_wxyz(ref["base_quat"].cpu().numpy()), hinges], axis=-1)
    q[:, :2] -= q[0, :2]                # recentre on the clip's first root xy
    return q


def render_pair(m, ren, cam, q_left, q_right, out, fps=60):
    d = mujoco.MjData(m)

    def frame(q):
        d.qpos[:] = q
        mujoco.mj_forward(m, d)
        cam.lookat[:] = [q[0], q[1], q[2] - 0.1]
        ren.update_scene(d, camera=cam)
        return ren.render()

    w = iio.get_writer(str(out), fps=fps, quality=8)
    for t in range(len(q_left)):
        combo = np.hstack([frame(q_left[t]), frame(q_right[t])])
        combo[:, ren.width - 2:ren.width + 2] = 30      # divider
        w.append_data(combo)
    w.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("clips", nargs="*", help="clip names (originals, without the mirror suffix)")
    ap.add_argument("--cache", default="cache/", help="preprocessed corpus dir")
    ap.add_argument("--grep", default=None, help="regex over clip names to select clips")
    ap.add_argument("--limit", type=int, default=4, help="max clips when using --grep")
    ap.add_argument("--out-dir", default="outputs/mirror")
    ap.add_argument("--azimuth", type=float, default=135.0, help="camera azimuth (deg)")
    ap.add_argument("--size", type=int, nargs=2, default=(480, 640), metavar=("H", "W"))
    args = ap.parse_args()

    corpus = MotionCorpus.load_cache(
        args.cache, output_device="cpu", urdf_hash=file_sha256(ROBOT_URDF),
        physics_fps=60, src_fps=30, keypoint_links=KEYPOINT_LINKS)
    names = list(args.clips)
    if args.grep:
        names += [n for n in corpus.clip_names if re.search(args.grep, n, re.I)][:args.limit]
    if not names:
        raise SystemExit("pass clip names or --grep")
    missing = [n for n in names if n not in corpus.clip_names]
    if missing:
        raise SystemExit(f"not in corpus: {missing}")
    corpus = mirror_corpus(corpus)

    m = mujoco.MjModel.from_xml_path(str(ROBOT_XML))
    ren = mujoco.Renderer(m, *args.size)
    cam = mujoco.MjvCamera()
    cam.azimuth, cam.elevation, cam.distance = args.azimuth, -10, 2.2

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    for name in names:
        q_l = clip_qpos(corpus, name, m.nq - 7)
        q_r = clip_qpos(corpus, name + MIRROR_SUFFIX, m.nq - 7)
        assert m.nq == q_l.shape[1], f"nq {m.nq} != qpos width {q_l.shape[1]}"
        safe = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_")
        out = out_dir / f"{safe}{MIRROR_SUFFIX}.mp4"
        render_pair(m, ren, cam, q_l, q_r, out)
        print(f"wrote {out} ({len(q_l)} frames, {len(q_l) / 60:.1f}s)  clip='{name}'")


if __name__ == "__main__":
    main()
