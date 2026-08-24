# tests/test_reward.py
import torch
from rgmt.env.reward import compute_reward, RewardWeights, RewardState, RewardRef

def _perfect(N=3, Kp=10):
    kp = torch.randn(N, Kp, 3)
    return (RewardState(kp_pos=kp, kp_vel=torch.zeros(N,Kp,3), root_pos=torch.zeros(N,3),
                        root_quat=torch.tensor([[0,0,0,1.0]]).repeat(N,1), root_h=torch.full((N,),0.63),
                        action=torch.zeros(N,23), prev_action=torch.zeros(N,23), fallen=torch.zeros(N,dtype=torch.bool)),
            RewardRef(kp_pos=kp, kp_vel=torch.zeros(N,Kp,3), root_pos=torch.zeros(N,3),
                      root_quat=torch.tensor([[0,0,0,1.0]]).repeat(N,1), root_h=torch.full((N,),0.63)))

def test_perfect_tracking_reward_is_high():
    s, r = _perfect()
    rew, terms = compute_reward(s, r, RewardWeights())
    assert terms["r_kp"].mean() > 0.99
    assert rew.shape == (3,)

def test_error_decreases_reward():
    s, r = _perfect()
    s2 = s
    s2.kp_pos = s.kp_pos + 0.5
    rew2, t2 = compute_reward(s2, r, RewardWeights())
    assert t2["r_kp"].mean() < 0.9
