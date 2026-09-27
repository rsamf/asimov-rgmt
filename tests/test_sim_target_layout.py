"""newton>=1.6 defaults Control.joint_target_q to the coord layout; NewtonSim
pins the legacy DOF layout for its own model without leaking the flag."""
import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="Newton sim needs CUDA")


def test_sim_pins_dof_layout_without_leaking_the_global_flag():
    import newton
    from rgmt.data.joint_map import KEYPOINT_LINKS
    from rgmt.env.sim import NewtonSim
    before = getattr(newton, "use_coord_layout_targets", None)
    sim = NewtonSim(1, kp=100.0, kd=5.0, control_decimation=1, dt=1 / 60, foot_friction=0.75,
                    keypoint_links=KEYPOINT_LINKS, device="cuda:0")
    assert getattr(newton, "use_coord_layout_targets", None) == before
    if before is not None:
        assert sim.model.use_coord_layout_targets is False
    assert len(sim.control.joint_target_q) == sim.nqd_per_env
