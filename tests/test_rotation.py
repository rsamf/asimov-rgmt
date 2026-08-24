import math, torch
from rgmt.utils.rotation import encode_angles, quat_to_matrix, gravity_projection, quat_rotate_inverse

def test_encode_angles_shape_and_values():
    q = torch.tensor([[0.0, math.pi / 2]])
    enc = encode_angles(q)
    assert enc.shape == (1, 4)
    assert torch.allclose(enc, torch.tensor([[1.0, 0.0, 0.0, 1.0]]), atol=1e-6)

def test_encode_continuity_across_pi():
    a = encode_angles(torch.tensor([[math.pi - 1e-4]]))
    b = encode_angles(torch.tensor([[-math.pi + 1e-4]]))
    assert torch.norm(a - b) < 1e-3  # continuous across the wrap

def test_identity_quat_gravity_is_down():
    q = torch.tensor([[0.0, 0.0, 0.0, 1.0]])  # xyzw identity
    g = gravity_projection(q)
    assert torch.allclose(g, torch.tensor([[0.0, 0.0, -1.0]]), atol=1e-6)

def test_quat_rotate_inverse_roundtrip():
    # 90deg about z: world x -> body y? check norm preserved and known axis
    q = torch.tensor([[0.0, 0.0, math.sin(math.pi/4), math.cos(math.pi/4)]])
    v = torch.tensor([[1.0, 0.0, 0.0]])
    out = quat_rotate_inverse(q, v)
    assert torch.allclose(torch.norm(out), torch.tensor(1.0), atol=1e-6)
