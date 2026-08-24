import torch
from rgmt.utils.pos_encoding import sinusoidal_pe

def test_shape_and_range():
    pe = sinusoidal_pe(21, 128)
    assert pe.shape == (21, 128)
    assert pe.abs().max() <= 1.0 + 1e-6

def test_distinct_rows():
    pe = sinusoidal_pe(10, 64)
    assert not torch.allclose(pe[0], pe[1])
