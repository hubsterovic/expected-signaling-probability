"""One Monte Carlo draw of the signaling probability, the transmission, the receiver's own-side sensitivity, or a pointwise ratio."""
from __future__ import annotations

import numpy as np
import qutip as qt

from expected_signaling_probability.core.config import DrawConfig, Quantity
from expected_signaling_probability.core.ensembles import draw_global_op, draw_local_op, draw_state


def child_rngs(master_seed: int, index: int, scheme: int) -> tuple[np.random.Generator, ...]:
    """Four independent generators for (state, second state, local op, global op).

    Every draw is identified by ``(master_seed, index)``; the scheme version is
    mixed in so that a change in the draw definition changes the streams too.
    """
    ss = np.random.SeedSequence([master_seed, index, scheme])
    return tuple(np.random.default_rng(child) for child in ss.spawn(4))


def _received_state(op, rho: qt.Qobj, receiver_index: int) -> qt.Qobj:
    return qt.ptrace(op(rho), receiver_index)


def draw(cfg: DrawConfig, master_seed: int, index: int) -> float:
    rng_state, rng_state_2, rng_local, rng_global = child_rngs(master_seed, index, cfg.scheme)

    rho = draw_state(cfg.state, cfg.d_A, cfg.d_B, rng_state)
    E = draw_global_op(cfg.global_op, cfg.d_A, cfg.d_B, rng_global, rank=cfg.global_rank, coupling=cfg.coupling)

    sigma = _received_state(E, rho, cfg.receiver_index)

    def signaling() -> float:
        A = draw_local_op(cfg.local_op, cfg.d_A, cfg.d_B, cfg.sender_index, rng_local)
        return float(qt.tracedist(sigma, _received_state(E, A(rho), cfg.receiver_index)))

    def local_sensitivity() -> float:
        # A fresh copy of the local-op stream, so that L is the same whether or not S was drawn first;
        # for d_A = d_B it is the sender's unitary applied on the receiver's side instead.
        rng_local_fresh = child_rngs(master_seed, index, cfg.scheme)[2]
        A = draw_local_op(cfg.local_op, cfg.d_A, cfg.d_B, cfg.receiver_index, rng_local_fresh)
        return float(qt.tracedist(sigma, _received_state(E, A(rho), cfg.receiver_index)))

    def transmission() -> float:
        rho_2 = draw_state(cfg.state, cfg.d_A, cfg.d_B, rng_state_2)
        return float(qt.tracedist(sigma, _received_state(E, rho_2, cfg.receiver_index)))

    match cfg.quantity:
        case Quantity.SIGNALING:
            return signaling()
        case Quantity.TRANSMISSION:
            return transmission()
        case Quantity.RATIO:
            return signaling() / transmission()
        case Quantity.LOCAL:
            return local_sensitivity()
        case Quantity.RATIO_LOCAL:
            return signaling() / local_sensitivity()
    raise ValueError(cfg.quantity)
