"""Random ensembles for states, local operations and global superoperators.

Every generator takes a ``numpy.random.Generator`` so that the three (or four)
random objects of one Monte Carlo draw can be given statistically independent
streams via ``SeedSequence.spawn``.

Operations are returned as plain callables ``rho -> rho`` so that unitary and
COMS maps can be applied as operators (cheap) while BCSZ maps are applied as
superoperators (expensive, but that is what they are).
"""
from __future__ import annotations

from collections.abc import Callable
from enum import Enum

import numpy as np
import qutip as qt

Rng = np.random.Generator
Operation = Callable[[qt.Qobj], qt.Qobj]


class StateEnsemble(str, Enum):
    """Ensemble from which the initial bipartite state rho_AB is drawn."""

    HAAR_PURE = "haar_pure"      # |psi> Haar random on H_A (x) H_B
    HS_MIXED = "hs_mixed"        # Hilbert-Schmidt measure (full-rank Ginibre) on H_A (x) H_B
    PRODUCT_HS = "product_hs"    # rho_A (x) rho_B, each Hilbert-Schmidt


class LocalOpEnsemble(str, Enum):
    """Ensemble for the sender's local operation A_X (x) id_Y."""

    HAAR_UNITARY = "haar_unitary"  # Haar random unitary on the sender's factor
    BCSZ = "bcsz"                  # full-rank BCSZ CPTP map on the sender's factor


class GlobalOpEnsemble(str, Enum):
    """Ensemble for the global superoperator E_AB."""

    HAAR_UNITARY = "haar_unitary"  # Haar random unitary on H_A (x) H_B (Kraus rank 1, erases nothing)
    COMS = "coms"                  # complete orthogonal measurement superoperator, Haar random basis of H_A (x) H_B (Beckman et al.)
    BCSZ = "bcsz"                  # BCSZ CPTP map on H_A (x) H_B, Kraus rank ``rank`` (None = full)
    WEAK_COUPLING = "weak"         # (U_A (x) U_B) exp(-i g H_AB), H_AB GUE with spectrum on [-2, 2]


# --------------------------------------------------------------------------- states

def draw_state(ensemble: StateEnsemble, d_A: int, d_B: int, rng: Rng) -> qt.Qobj:
    dims = [d_A, d_B]
    match ensemble:
        case StateEnsemble.HAAR_PURE:
            return qt.rand_ket(dims, seed=rng).proj()
        case StateEnsemble.HS_MIXED:
            return qt.rand_dm(dims, distribution="hs", seed=rng)
        case StateEnsemble.PRODUCT_HS:
            rho_A = qt.rand_dm(d_A, distribution="hs", seed=rng)
            rho_B = qt.rand_dm(d_B, distribution="hs", seed=rng)
            return qt.tensor(rho_A, rho_B)
    raise ValueError(ensemble)


# --------------------------------------------------------------------------- local ops

def _embed_operator(op_sender: qt.Qobj, d_A: int, d_B: int, sender_index: int) -> qt.Qobj:
    if sender_index == 0:
        return qt.tensor(op_sender, qt.identity(d_B))
    return qt.tensor(qt.identity(d_A), op_sender)


def _embed_superoperator(sop_sender: qt.Qobj, d_A: int, d_B: int, sender_index: int) -> qt.Qobj:
    if sender_index == 0:
        return qt.super_tensor(sop_sender, qt.to_super(qt.identity(d_B)))
    return qt.super_tensor(qt.to_super(qt.identity(d_A)), sop_sender)


def draw_local_op(ensemble: LocalOpEnsemble, d_A: int, d_B: int, sender_index: int, rng: Rng) -> Operation:
    """Local operation acting on the sender's factor only (``sender_index`` 0 = A, 1 = B)."""
    d_sender = (d_A, d_B)[sender_index]
    match ensemble:
        case LocalOpEnsemble.HAAR_UNITARY:
            U = _embed_operator(qt.rand_unitary(d_sender, seed=rng), d_A, d_B, sender_index)
            U_dag = U.dag()
            return lambda rho: U * rho * U_dag
        case LocalOpEnsemble.BCSZ:
            sop = _embed_superoperator(qt.rand_super_bcsz(d_sender, seed=rng), d_A, d_B, sender_index)
            return lambda rho: sop(rho)
    raise ValueError(ensemble)


# --------------------------------------------------------------------------- global ops

def _coms_map(U: qt.Qobj) -> Operation:
    """rho -> sum_k |u_k><u_k| rho |u_k><u_k| for the basis {|u_k>} = columns of U."""
    U_dag = U.dag()

    def apply(rho: qt.Qobj) -> qt.Qobj:
        M = (U_dag * rho * U).full()
        D = qt.Qobj(np.diag(np.diag(M)), dims=rho.dims)
        return U * D * U_dag

    return apply


def _gue(D: int, rng: Rng) -> np.ndarray:
    """GUE matrix normalized so the spectrum converges to the semicircle on [-2, 2]."""
    G = rng.normal(size=(D, D)) + 1j * rng.normal(size=(D, D))
    return (G + G.conj().T) / (2 * np.sqrt(D))


def draw_global_op(ensemble: GlobalOpEnsemble, d_A: int, d_B: int, rng: Rng, rank: int | None = None, coupling: float | None = None) -> Operation:
    dims = [d_A, d_B]
    match ensemble:
        case GlobalOpEnsemble.WEAK_COUPLING:
            assert coupling is not None
            from scipy.linalg import expm

            U_loc = qt.tensor(qt.rand_unitary(d_A, seed=rng), qt.rand_unitary(d_B, seed=rng))
            V = qt.Qobj(expm(-1j * coupling * _gue(d_A * d_B, rng)), dims=U_loc.dims)
            U = U_loc * V
            U_dag = U.dag()
            return lambda rho: U * rho * U_dag
        case GlobalOpEnsemble.HAAR_UNITARY:
            U = qt.rand_unitary(dims, seed=rng)
            U_dag = U.dag()
            return lambda rho: U * rho * U_dag
        case GlobalOpEnsemble.COMS:
            return _coms_map(qt.rand_unitary(dims, seed=rng))
        case GlobalOpEnsemble.BCSZ:
            sop = qt.rand_super_bcsz(dims, rank=rank, seed=rng)
            return lambda rho: sop(rho)
    raise ValueError(ensemble)
