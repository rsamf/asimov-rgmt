"""Throughput probe: where does a training iteration spend its time?

Measures env.step decomposition (sim vs bundle/corpus-gather) and policy
act/update time at several env counts. Short (~64 steps each) — a profiling
probe, not a training run.
"""
import sys
import time

import torch

from rgmt.data.corpus import MotionCorpus
from rgmt.data.cache_key import file_sha256
from rgmt.data.joint_map import KEYPOINT_LINKS
from rgmt.assets.paths import ROBOT_URDF
from rgmt.env.track_env import TrackEnv, EnvConfig
from rgmt.policy.networks import RGMTActorCritic, PolicyDims
from rgmt.algos.rollout import RolloutBuffer
from rgmt.algos.ppo import PPOTrainer, PPOConfig

CACHE = "cache/"
DEV = "cuda:0"
STEPS = 32


def sync():
    torch.cuda.synchronize()


def profile(n_envs: int, corpus: MotionCorpus):
    torch.cuda.reset_peak_memory_stats()
    cfg = EnvConfig(num_envs=n_envs, keypoint_links=KEYPOINT_LINKS, episode_len=1000)
    env = TrackEnv(cfg, corpus, device=DEV)
    dims = PolicyDims(priv_dim=env.priv_dim)
    model = RGMTActorCritic(dims).to(DEV)
    ppo = PPOTrainer(model, PPOConfig(n_epochs=5, mb_size=16384))
    buf = RolloutBuffer(n_envs, STEPS, dims, DEV)

    # instrument sim.step and the command-window path
    t_sim = t_cmd = 0.0
    real_sim_step = env.sim.step
    real_cmdwin = env._command_window_noisy

    def timed_sim(*a, **k):
        nonlocal t_sim
        sync(); t0 = time.perf_counter()
        out = real_sim_step(*a, **k)
        sync(); t_sim += time.perf_counter() - t0
        return out

    def timed_cmd(*a, **k):
        nonlocal t_cmd
        sync(); t0 = time.perf_counter()
        out = real_cmdwin(*a, **k)
        sync(); t_cmd += time.perf_counter() - t0
        return out

    env.sim.step = timed_sim
    env._command_window_noisy = timed_cmd

    bundle = env.reset_all()
    for _ in range(8):  # warmup
        with torch.no_grad():
            a, _, _ = model.act(bundle)
        bundle, *_ = env.step(a)
    t_sim = t_cmd = 0.0

    t_act = t_env = 0.0
    sync(); t_roll0 = time.perf_counter()
    for _ in range(STEPS):
        sync(); t0 = time.perf_counter()
        with torch.no_grad():
            a, logp, v = model.act(bundle)
        sync(); t_act += time.perf_counter() - t0
        t0 = time.perf_counter()
        nxt, rew, done, info = env.step(a)
        sync(); t_env += time.perf_counter() - t0
        buf.add(bundle, a, logp, v, rew, done)
        bundle = nxt
    sync()
    t_roll = time.perf_counter() - t_roll0

    with torch.no_grad():
        last_v = model.value(bundle)
    data = buf.compute_gae(last_v, 0.99, 0.95)
    # PPO update needs T=32 slots; we filled 64 adds into 32-slot buffer twice over —
    # RolloutBuffer wraps by clear(); to keep this probe simple just time one update
    # on whatever compute_gae returned.
    sync(); t0 = time.perf_counter()
    ppo.update(data)
    sync(); t_ppo = time.perf_counter() - t0

    eps = n_envs * STEPS / t_roll
    other = t_env - t_sim - t_cmd
    print(f"N={n_envs:5d} | {eps/1e3:7.1f}k env-steps/s | "
          f"step: sim {t_sim/STEPS*1e3:6.2f}ms  cmd-win {t_cmd/STEPS*1e3:6.2f}ms  "
          f"other-env {other/STEPS*1e3:6.2f}ms  policy {t_act/STEPS*1e3:6.2f}ms | "
          f"ppo/update {t_ppo:5.2f}s | peakGPU {torch.cuda.max_memory_allocated()/1e9:.2f}GB")
    del env, model, ppo, buf, data
    torch.cuda.empty_cache()


if __name__ == "__main__":
    corpus = MotionCorpus.load_cache(
        CACHE, output_device=DEV, urdf_hash=file_sha256(ROBOT_URDF),
        physics_fps=60, src_fps=30, keypoint_links=KEYPOINT_LINKS)
    args = [a for a in sys.argv[1:] if a != "--gpu-corpus"]
    if "--gpu-corpus" in sys.argv:
        corpus.to_storage(DEV); print("corpus storage: GPU")
    else:
        print("corpus storage: CPU")
    sizes = [int(s) for s in args] or [2048]
    for n in sizes:
        profile(n, corpus)
