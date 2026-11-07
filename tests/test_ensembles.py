import numpy as np
import pytest
import qutip as qt

from expected_signaling_probability.core.ensembles import (
    GlobalOpEnsemble,
    LocalOpEnsemble,
    StateEnsemble,
    draw_global_op,
    draw_local_op,
    draw_state,
)

DIMS = [(2, 2), (3, 2), (2, 4), (3, 3)]


def _is_density_matrix(rho: qt.Qobj, tol=1e-9) -> bool:
    return abs(rho.tr() - 1) < tol and rho.isherm and np.all(rho.eigenenergies() > -tol)


@pytest.mark.parametrize("ensemble", list(StateEnsemble))
@pytest.mark.parametrize("d_A,d_B", DIMS)
def test_states_are_density_matrices_with_right_dims(ensemble, d_A, d_B):
    rho = draw_state(ensemble, d_A, d_B, np.random.default_rng(1))
    assert rho.dims == [[d_A, d_B], [d_A, d_B]]
    assert _is_density_matrix(rho)


def test_haar_pure_is_pure_and_product_is_product():
    rng = np.random.default_rng(2)
    rho = draw_state(StateEnsemble.HAAR_PURE, 3, 2, rng)
    assert abs((rho * rho).tr() - 1) < 1e-9
    rho = draw_state(StateEnsemble.PRODUCT_HS, 3, 2, rng)
    assert (rho - qt.tensor(rho.ptrace(0), rho.ptrace(1))).norm() < 1e-9


@pytest.mark.parametrize("ensemble", list(LocalOpEnsemble))
@pytest.mark.parametrize("sender_index", [0, 1])
@pytest.mark.parametrize("d_A,d_B", DIMS)
def test_local_op_does_not_change_receiver_reduced_state(ensemble, sender_index, d_A, d_B):
    rng = np.random.default_rng(3)
    rho = draw_state(StateEnsemble.HAAR_PURE, d_A, d_B, rng)
    A = draw_local_op(ensemble, d_A, d_B, sender_index, rng)
    out = A(rho)
    assert _is_density_matrix(out)
    receiver = 1 - sender_index
    assert qt.tracedist(rho.ptrace(receiver), out.ptrace(receiver)) < 1e-9
    # and it does act on the sender
    assert qt.tracedist(rho.ptrace(sender_index), out.ptrace(sender_index)) > 1e-3


@pytest.mark.parametrize("ensemble", list(GlobalOpEnsemble))
@pytest.mark.parametrize("d_A,d_B", DIMS)
def test_global_op_is_trace_preserving_and_positive(ensemble, d_A, d_B):
    rng = np.random.default_rng(4)
    rho = draw_state(StateEnsemble.HS_MIXED, d_A, d_B, rng)
    E = draw_global_op(ensemble, d_A, d_B, rng, coupling=1.0)
    out = E(rho)
    assert out.dims == rho.dims
    assert _is_density_matrix(out)


def test_coms_is_idempotent_and_unitary_preserves_purity():
    rng = np.random.default_rng(5)
    rho = draw_state(StateEnsemble.HAAR_PURE, 3, 2, rng)
    L = draw_global_op(GlobalOpEnsemble.COMS, 3, 2, rng)
    once, twice = L(rho), L(L(rho))
    assert (once - twice).norm() < 1e-9
    assert (once * once).tr() < 1 - 1e-3  # a measurement decoheres a generic pure state
    U = draw_global_op(GlobalOpEnsemble.HAAR_UNITARY, 3, 2, rng)
    assert abs((U(rho) * U(rho)).tr() - 1) < 1e-9


def test_bcsz_rank_is_honoured():
    rng = np.random.default_rng(6)
    sop = qt.rand_super_bcsz([2, 2], rank=1, seed=rng)
    kraus = qt.to_kraus(sop)
    assert sum(k.norm() > 1e-8 for k in kraus) == 1
