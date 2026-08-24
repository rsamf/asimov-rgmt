"""Annealed upward assist-force schedule (pure-CPU logic test)."""


def test_linear_anneal_schedule():
    from rgmt.train import assist_scale_at

    assert assist_scale_at(0, total=100) == 1.0
    assert abs(assist_scale_at(50, total=100) - 0.5) < 1e-6
    assert assist_scale_at(100, total=100) == 0.0
    assert assist_scale_at(200, total=100) == 0.0
