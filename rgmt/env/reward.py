from dataclasses import dataclass, field
import torch
from torch import Tensor

@dataclass
class RewardState:
    kp_pos: Tensor; kp_vel: Tensor; root_pos: Tensor; root_quat: Tensor; root_h: Tensor
    action: Tensor; prev_action: Tensor; fallen: Tensor

@dataclass
class RewardRef:
    kp_pos: Tensor; kp_vel: Tensor; root_pos: Tensor; root_quat: Tensor; root_h: Tensor

@dataclass
class RewardWeights:
    # Sharpness raised after eval showed a degenerate stander: at s_kp=1.0 a 34cm
    # keypoint error scored ~0.96 (near-perfect), so standing still was a reward
    # optimum. Sharper kernels give real gradient pressure to track.
    w_kp: float = 1.0; s_kp: float = 25.0
    w_rel: float = 0.3; s_rel: float = 25.0
    # Velocity kernel softened (2.0 -> 0.4) + upweighted (0.1 -> 0.3):
    # r_kpv previously sat at 0.000 (kernel too sharp = zero gradient), so
    # nothing distinguished moving-with-the-reference from hovering near it.
    w_kpv: float = 0.3; s_kpv: float = 0.4
    w_rh: float = 0.2; s_rh: float = 50.0
    w_rp: float = 0.05; s_rp: float = 20.0
    w_rq: float = 0.15; s_rq: float = 8.0
    w_action: float = 0.005
    w_arate: float = 0.01
    w_alive: float = 0.05
    w_fall: float = 1.0

def _root_relative(kp, root):
    return kp - root.unsqueeze(1)

def compute_reward(s: RewardState, r: RewardRef, w: RewardWeights):
    r_kp = torch.exp(-w.s_kp * ((s.kp_pos - r.kp_pos) ** 2).flatten(1).mean(1))
    rel = _root_relative(s.kp_pos, s.root_pos) - _root_relative(r.kp_pos, r.root_pos)
    r_rel = torch.exp(-w.s_rel * (rel ** 2).flatten(1).mean(1))
    r_kpv = torch.exp(-w.s_kpv * ((s.kp_vel - r.kp_vel) ** 2).flatten(1).mean(1))
    r_rh = torch.exp(-w.s_rh * (s.root_h - r.root_h) ** 2)
    r_rp = torch.exp(-w.s_rp * ((s.root_pos[:, :2] - r.root_pos[:, :2]) ** 2).sum(1))
    # quaternion geodesic angle
    dot = (s.root_quat * r.root_quat).sum(-1).abs().clamp(max=1.0)
    ang = 2 * torch.acos(dot)
    r_rq = torch.exp(-w.s_rq * ang ** 2)
    # Joint-exponent (multiplicative) tracking = prod_i r_i^{w_i}
    #   = exp(-sum_i w_i * s_i * e_i).
    # Under the previous additive sum each kernel saturated independently, so
    # the policy cherry-picked easy terms (r_rel, r_rh) and ignored hard ones
    # (r_kpv stuck ~0.05 for whole runs). One shared exponent keeps gradient flowing
    # through the WORST term at all times. Scaled by 2.0 so tracking spans
    # ~[0, 2] and clearly dominates alive(0.05)/penalties near the reference.
    tracking = 2.0 * (
        r_kp.clamp_min(1e-8) ** w.w_kp
        * r_rel.clamp_min(1e-8) ** w.w_rel
        * r_kpv.clamp_min(1e-8) ** w.w_kpv
        * r_rh.clamp_min(1e-8) ** w.w_rh
        * r_rp.clamp_min(1e-8) ** w.w_rp
        * r_rq.clamp_min(1e-8) ** w.w_rq
    )
    act_pen = w.w_action * (s.action ** 2).mean(1)
    arate_pen = w.w_arate * ((s.action - s.prev_action) ** 2).mean(1)
    alive = w.w_alive * (~s.fallen).float()
    fall_pen = w.w_fall * s.fallen.float()
    reward = tracking - act_pen - arate_pen + alive - fall_pen
    terms = dict(r_kp=r_kp, r_rel=r_rel, r_kpv=r_kpv, r_rh=r_rh, r_rp=r_rp, r_rq=r_rq,
                 act_pen=act_pen, arate_pen=arate_pen, alive=alive, fall_pen=fall_pen,
                 tracking=tracking)
    return reward, terms
