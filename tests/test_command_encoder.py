import torch
from rgmt.policy.encoders import CommandEncoder

def test_output_shape():
    enc = CommandEncoder(cmd_dim=55, n_embd=128)
    u = enc(torch.randn(4, 128), torch.randn(4, 21, 55))
    assert u.shape == (4, 128)

def test_depends_on_command_window():
    torch.manual_seed(0)
    enc = CommandEncoder(cmd_dim=55, n_embd=128).eval()
    h = torch.randn(2, 128); c = torch.randn(2, 21, 55)
    u1 = enc(h, c); u2 = enc(h, c + 1.0)
    assert not torch.allclose(u1, u2)
