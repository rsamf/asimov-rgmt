import math, torch
from torch import Tensor

def sinusoidal_pe(seq_len: int, dim: int, device=None, dtype=None) -> Tensor:
    pe = torch.zeros(seq_len, dim, device=device, dtype=dtype or torch.float32)
    pos = torch.arange(seq_len, device=device, dtype=pe.dtype).unsqueeze(1)
    i = torch.arange(0, dim, 2, device=device, dtype=pe.dtype)
    div = torch.exp(-math.log(10000.0) * i / dim)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)[:, : pe[:, 1::2].shape[1]]
    return pe
