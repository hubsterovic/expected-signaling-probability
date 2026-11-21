"""Exploration: Frobenius distance of a random channel's Choi matrix to the
linear subspace of A->B semicausal channels, normalised by the distance to the
completely depolarizing channel.  Lower bound on the distance to the nearest
causal channel (positivity is ignored)."""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import qutip as qt

from expected_signaling_probability.core.config import GlobalOpEnsemble
from expected_signaling_probability.core.draw import child_rngs
from expected_signaling_probability.core.ensembles import draw_global_op


def choi(op, d_A: int, d_B: int) -> np.ndarray:
    """J[(i,k),(j,l)] = <k| E(|i><j|) |l>, index order (in, out)."""
    D = d_A * d_B
    J = np.empty((D, D, D, D), dtype=complex)
    for i in range(D):
        for j in range(D):
            E = np.zeros((D, D), dtype=complex)
            E[i, j] = 1.0
            J[i, :, j, :] = op(qt.Qobj(E, dims=[[d_A, d_B], [d_A, d_B]])).full()
    return J.reshape(D * D, D * D)


def acausal_part_sq(J: np.ndarray, d_A: int, d_B: int, sender: int) -> float:
    """||J - P_causal(J)||_F^2 where P_causal projects onto the subspace in
    which the receiver's output does not depend on the sender's input."""
    d = (d_A, d_B)
    d_s, d_r = d[sender], d[1 - sender]
    # indices: s_in, r_in, s_out, r_out (and primed) in the order (A, B)
    J8 = J.reshape(d_A, d_B, d_A, d_B, d_A, d_B, d_A, d_B)
    if sender == 1:  # swap A and B roles
        J8 = J8.transpose(1, 0, 3, 2, 5, 4, 7, 6)
    # trace out the sender's output
    M = np.trace(J8, axis1=2, axis2=6)  # -> s_in, r_in, r_out, s_in', r_in', r_out'
    # M has axes (s_in, r_in, r_out, s_in', r_in', r_out'); its 1_s ⊗ K part:
    K = np.trace(M, axis1=0, axis2=3) / d_s  # (r_in, r_out, r_in', r_out')
    M_perp = M - np.einsum("sS,abAB->sabSAB", np.eye(d_s), K)
    return float(np.sum(np.abs(M_perp) ** 2) / d_s)


def normalised_distance(J: np.ndarray, d_A: int, d_B: int, sender: int) -> float:
    D = d_A * d_B
    J0 = np.eye(D * D) / D
    return np.sqrt(acausal_part_sq(J, d_A, d_B, sender)) / np.linalg.norm(J - J0)


def unitary_op(U: np.ndarray, d_A: int, d_B: int):
    Uq = qt.Qobj(U, dims=[[d_A, d_B], [d_A, d_B]])
    return lambda rho: Uq * rho * Uq.dag()


def sanity() -> None:
    swap = np.zeros((4, 4))
    for a in range(2):
        for b in range(2):
            swap[2 * b + a, 2 * a + b] = 1
    cnot = np.eye(4)[[0, 1, 3, 2]]
    prod = np.kron(qt.rand_unitary(2, seed=1).full(), qt.rand_unitary(2, seed=2).full())
    for name, U in [("SWAP", swap), ("CNOT", cnot), ("product", prod), ("identity", np.eye(4))]:
        J = choi(unitary_op(U, 2, 2), 2, 2)
        print(f"{name:9s} r(A->B)={normalised_distance(J, 2, 2, 0):.3f} r(B->A)={normalised_distance(J, 2, 2, 1):.3f}")
    # check the Choi convention: E(X) = sum_ij X_ij E(|i><j|)
    op = draw_global_op(GlobalOpEnsemble.BCSZ, 2, 3, np.random.default_rng(0))
    J = choi(op, 2, 3)
    X = qt.rand_dm([2, 3], seed=3)
    lhs = op(X).full()
    rhs = np.einsum("ij,ikjl->kl", X.full(), J.reshape(6, 6, 6, 6))
    print("choi convention ok:", np.allclose(lhs, rhs))


def run(n_samples: int) -> pd.DataFrame:
    rows = []
    configs = []
    for ens, d_max in [(GlobalOpEnsemble.HAAR_UNITARY, 8), (GlobalOpEnsemble.COMS, 8), (GlobalOpEnsemble.BCSZ, 6)]:
        configs += [(ens, d, d) for d in range(2, d_max + 1)]
    for ens, dA_max in [(GlobalOpEnsemble.HAAR_UNITARY, 12), (GlobalOpEnsemble.COMS, 12), (GlobalOpEnsemble.BCSZ, 8)]:
        configs += [(ens, d_A, 2) for d_A in range(2, dA_max + 1)]
    for ens, d_A, d_B in configs:
        for i in range(n_samples):
            rng = child_rngs(0, i, 0)[0]
            J = choi(draw_global_op(ens, d_A, d_B, rng), d_A, d_B)
            for sender, name in [(0, "AtoB"), (1, "BtoA")]:
                rows.append(dict(ensemble=ens.value, d_A=d_A, d_B=d_B, direction=name, i=i,
                                 r=normalised_distance(J, d_A, d_B, sender),
                                 delta_sq=acausal_part_sq(J, d_A, d_B, sender)))
        df = pd.DataFrame([r for r in rows if r["ensemble"] == ens.value and r["d_A"] == d_A and r["d_B"] == d_B])
        med = df.groupby("direction")["r"].median()
        print(f"{ens.value:12s} d_A={d_A:2d} d_B={d_B:2d}  r(A->B)={med['AtoB']:.3f}  r(B->A)={med['BtoA']:.3f}", flush=True)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    sanity()
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    df = run(n)
    df.to_csv("data/explore/causal_distance.csv", index=False)
