import numpy as np
import pytest
import qutip as qt

from expected_signaling_probability import Direction, DrawConfig, GlobalOpEnsemble, Quantity, draw
from expected_signaling_probability.core.draw import child_rngs


def test_draw_is_deterministic_and_index_dependent():
    cfg = DrawConfig(d_A=3, d_B=2)
    assert draw(cfg, 0, 0) == draw(cfg, 0, 0)
    assert draw(cfg, 0, 0) != draw(cfg, 0, 1)
    assert draw(cfg, 0, 0) != draw(cfg, 1, 0)


def test_child_streams_are_independent():
    rngs = child_rngs(master_seed=0, index=0, scheme=2)
    draws = [r.normal(size=8) for r in rngs]
    for i in range(len(draws)):
        for j in range(i + 1, len(draws)):
            assert not np.allclose(draws[i], draws[j])


def test_scheme_changes_streams():
    a = child_rngs(0, 0, 1)[0].normal(size=4)
    b = child_rngs(0, 0, 2)[0].normal(size=4)
    assert not np.allclose(a, b)


@pytest.mark.parametrize("direction", list(Direction))
@pytest.mark.parametrize("quantity", list(Quantity))
@pytest.mark.parametrize("global_op", list(GlobalOpEnsemble))
def test_draw_is_a_probability(direction, quantity, global_op):
    cfg = DrawConfig(d_A=3, d_B=2, direction=direction, quantity=quantity, global_op=global_op, coupling=1.0 if global_op is GlobalOpEnsemble.WEAK_COUPLING else None)
    v = draw(cfg, 0, 0)
    assert v >= 0.0
    if quantity is not Quantity.RATIO:
        assert v <= 1.0


def test_config_validation_and_slug():
    with pytest.raises(ValueError):
        DrawConfig(d_A=1, d_B=2)
    with pytest.raises(ValueError):
        DrawConfig(d_A=2, d_B=2, global_rank=3)  # rank only for BCSZ
    cfg = DrawConfig(d_A=4, d_B=2, global_op=GlobalOpEnsemble.BCSZ, global_rank=3)
    assert cfg.slug() == "S_dA=4_dB=2_AtoB_state=haar_pure_local=haar_unitary_global=bcsz_rank=3_v2"
    assert cfg.with_(d_A=5).d_A == 5 and cfg.d_A == 4


def test_signaling_vanishes_for_a_causal_global_op(monkeypatch):
    """Patch the global op to be a product unitary; then S must be exactly 0."""
    import expected_signaling_probability.core.draw as draw_mod

    def product_unitary(ensemble, d_A, d_B, rng, rank=None, coupling=None):
        U = qt.tensor(qt.rand_unitary(d_A, seed=rng), qt.rand_unitary(d_B, seed=rng))
        return lambda rho: U * rho * U.dag()

    monkeypatch.setattr(draw_mod, "draw_global_op", product_unitary)
    for direction in Direction:
        assert draw(DrawConfig(d_A=3, d_B=2, direction=direction), 0, 0) < 1e-9


def test_ratio_matches_pointwise_signaling_over_transmission():
    base = DrawConfig(d_A=3, d_B=2, global_op=GlobalOpEnsemble.COMS)
    s = draw(base, 0, 4)
    t = draw(base.with_(quantity=Quantity.TRANSMISSION), 0, 4)
    r = draw(base.with_(quantity=Quantity.RATIO), 0, 4)
    assert abs(r - s / t) < 1e-12


def test_weak_coupling_is_causal_at_zero_coupling():
    cfg = DrawConfig(d_A=3, d_B=2, global_op=GlobalOpEnsemble.WEAK_COUPLING, coupling=0.0)
    assert draw(cfg, 0, 0) < 1e-9
    assert draw(cfg.with_(quantity=Quantity.TRANSMISSION), 0, 0) > 1e-3
