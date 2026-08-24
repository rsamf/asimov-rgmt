import torch
from rgmt.algos.rollout import RolloutBuffer
from rgmt.policy.networks import PolicyDims


def test_gae_masks_on_done():
    """GAE done=1 at t=1 must zero the cross-boundary bootstrap.

    Setup: 1 env, 3 steps, done at t=1 only, constant reward=1,
    value=0.5 everywhere, last_value=0.5, gamma=0.99, lam=0.95.

    Recursion (reversed):
      t=2: non_terminal=1, delta=1+0.99*0.5-0.5=0.995, gae=0.995             → A_2=0.995
      t=1: non_terminal=0, delta=1+0-0.5=0.5,           gae=0.5+0*...=0.5    → A_1=0.5
      t=0: non_terminal=1, delta=0.995,                  gae=0.995+0.99*0.95*0.5=1.46525 → A_0≈1.465
    """
    d = PolicyDims(priv_dim=8)
    buf = RolloutBuffer(num_envs=1, rollout_len=3, dims=d, device="cpu")

    def bundle():
        return dict(
            obs=torch.zeros(1, d.obs_dim),
            history=torch.zeros(1, d.hist_len, d.obs_dim),
            cmd_window=torch.zeros(1, d.cmd_len, d.cmd_dim),
            critic_obs=torch.zeros(1, d.priv_dim),
        )

    dones = [0.0, 1.0, 0.0]
    for done_val in dones:
        buf.add(
            bundle(),
            torch.zeros(1, 23),
            torch.zeros(1),
            torch.tensor([0.5]),
            torch.tensor([1.0]),
            torch.tensor([done_val]),
        )

    data = buf.compute_gae(last_value=torch.tensor([0.5]), gamma=0.99, lam=0.95)

    # Hand-computed from the same recursion the code uses:
    gamma, lam = 0.99, 0.95
    gl = gamma * lam
    # t=2: non_terminal=1, delta=1+gamma*0.5-0.5=0.995, gae=0.995
    a2 = 0.995
    # t=1: non_terminal=0 (done=1), delta=1+0-0.5=0.5, gae=0.5+gl*0*a2=0.5
    a1 = 0.5
    # t=0: non_terminal=1 (done=0), delta=1+gamma*0.5-0.5=0.995, gae=0.995+gl*1*a1
    a0 = 0.995 + gl * a1

    expected = torch.tensor([a0, a1, a2])
    assert torch.allclose(data["advantages"], expected, atol=1e-3), (
        f"Expected {expected.tolist()}, got {data['advantages'].tolist()}"
    )


def test_gae_matches_hand_computation():
    d = PolicyDims(priv_dim=8)
    buf = RolloutBuffer(num_envs=1, rollout_len=3, dims=d, device="cpu")
    def bundle():
        return dict(obs=torch.zeros(1, d.obs_dim), history=torch.zeros(1, d.hist_len, d.obs_dim),
                    cmd_window=torch.zeros(1, d.cmd_len, d.cmd_dim), critic_obs=torch.zeros(1, d.priv_dim))
    rewards = [1.0, 1.0, 1.0]; values = [0.5, 0.5, 0.5]
    for rr, vv in zip(rewards, values):
        buf.add(bundle(), torch.zeros(1, 23), torch.zeros(1), torch.tensor([vv]),
                torch.tensor([rr]), torch.tensor([0.0]))
    data = buf.compute_gae(last_value=torch.tensor([0.5]), gamma=0.99, lam=0.95)
    # delta_t = r + gamma*v_{t+1} - v_t = 1 + 0.99*0.5 - 0.5 = 0.995 each
    # A_2=0.995; A_1=0.995+gl*A_2; A_0=0.995+gl*A_1
    gl = 0.99 * 0.95
    a2 = 0.995; a1 = 0.995 + gl * a2; a0 = 0.995 + gl * a1
    assert torch.allclose(data["advantages"], torch.tensor([a0, a1, a2]), atol=1e-4)
