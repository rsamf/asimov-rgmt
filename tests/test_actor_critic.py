# tests/test_actor_critic.py
import torch
from rgmt.policy.networks import RGMTActorCritic, PolicyDims

def _bundle(B, d):
    return dict(obs=torch.randn(B, d.obs_dim), history=torch.randn(B, d.hist_len, d.obs_dim),
                cmd_window=torch.randn(B, d.cmd_len, d.cmd_dim), critic_obs=torch.randn(B, d.priv_dim))

def test_shapes_and_consistency():
    d = PolicyDims(priv_dim=64)
    ac = RGMTActorCritic(d)
    b = _bundle(8, d)
    a, lp, v = ac.act(b)
    assert a.shape == (8, 23) and lp.shape == (8,) and v.shape == (8,)
    lp2, ent, v2 = ac.evaluate(b, a)
    assert lp2.shape == (8,) and ent.shape == (8,) and v2.shape == (8,)
    assert torch.allclose(v, v2, atol=1e-5)

def test_inference_is_deterministic():
    d = PolicyDims(priv_dim=64)
    ac = RGMTActorCritic(d).eval()
    b = _bundle(4, d)
    assert torch.allclose(ac.act_inference(b), ac.act_inference(b))
