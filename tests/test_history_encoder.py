import torch
from rgmt.policy.encoders import HistoryEncoder

def test_output_shape():
    enc = HistoryEncoder(in_dim=98, n_embd=128)
    h = enc(torch.randn(4, 10, 98))
    assert h.shape == (4, 128)

def test_causality_last_token_change_propagates_but_not_backward():
    torch.manual_seed(0)
    enc = HistoryEncoder(in_dim=8, n_embd=16).eval()
    x = torch.randn(1, 5, 8)
    # perturb only the FIRST timestep; with causal mask + maxpool, later per-token
    # features for t>=1 must be unaffected at their own positions is hard to assert post-pool;
    # instead assert pre-pool token 0 is independent of future tokens:
    base = enc.token_features(x)              # (1,5,16) pre-maxpool
    x2 = x.clone(); x2[:, 4] += 5.0           # perturb LAST token
    pert = enc.token_features(x2)
    assert torch.allclose(base[:, 0], pert[:, 0], atol=1e-5)  # token0 cannot see token4
    assert not torch.allclose(base[:, 4], pert[:, 4], atol=1e-5)
