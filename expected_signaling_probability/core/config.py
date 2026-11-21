"""A ``DrawConfig`` fully declares one Monte Carlo draw (up to the random seed)."""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from enum import Enum
from typing import Any

from expected_signaling_probability.core.ensembles import GlobalOpEnsemble, LocalOpEnsemble, StateEnsemble
from expected_signaling_probability.utils.directions import Direction

# Bump this whenever the meaning of a draw changes (seeding scheme, ensemble
# definitions, ...). It is part of every cache key, so old caches are never
# silently reused.
SCHEME_VERSION = 2


class Quantity(str, Enum):
    SIGNALING = "S"     # D_tr( Tr_X E(rho), Tr_X E(A_X(rho)) )
    TRANSMISSION = "T"  # D_tr( Tr_X E(rho_1), Tr_X E(rho_2) ), rho_1, rho_2 independent; erasure baseline
    RATIO = "R"         # S / T evaluated pointwise on the same (rho, E); report the median
    LOCAL = "L"         # D_tr( Tr_X E(rho), Tr_X E(A_X(rho)) ) with the local op on the RECEIVER X: the receiver's own-side sensitivity
    RATIO_LOCAL = "RL"  # S / L evaluated pointwise on the same (rho, E); 1 means Alice's intervention is as visible to Bob as his own


@dataclass(frozen=True)
class DrawConfig:
    d_A: int
    d_B: int
    direction: Direction = Direction.A_TO_B
    state: StateEnsemble = StateEnsemble.HAAR_PURE
    local_op: LocalOpEnsemble = LocalOpEnsemble.HAAR_UNITARY
    global_op: GlobalOpEnsemble = GlobalOpEnsemble.HAAR_UNITARY
    global_rank: int | None = None  # Kraus rank for BCSZ global ops; None = full rank
    coupling: float | None = None   # interaction strength g for WEAK_COUPLING global ops
    quantity: Quantity = Quantity.SIGNALING
    scheme: int = SCHEME_VERSION

    def __post_init__(self) -> None:
        if self.d_A < 2 or self.d_B < 2:
            raise ValueError("subsystem dimensions must be >= 2")
        if self.global_rank is not None and self.global_op is not GlobalOpEnsemble.BCSZ:
            raise ValueError("global_rank is only meaningful for GlobalOpEnsemble.BCSZ")
        if (self.coupling is None) != (self.global_op is not GlobalOpEnsemble.WEAK_COUPLING):
            raise ValueError("coupling must be given exactly when global_op is GlobalOpEnsemble.WEAK_COUPLING")

    @property
    def sender_index(self) -> int:
        return 0 if self.direction is Direction.A_TO_B else 1

    @property
    def receiver_index(self) -> int:
        return 1 - self.sender_index

    def with_(self, **changes: Any) -> "DrawConfig":
        return replace(self, **changes)

    def slug(self) -> str:
        rank = "full" if self.global_rank is None else str(self.global_rank)
        return "_".join(
            [
                self.quantity.value,
                f"dA={self.d_A}",
                f"dB={self.d_B}",
                self.direction.to_str(),
                f"state={self.state.value}",
                f"local={self.local_op.value}",
                f"global={self.global_op.value}",
                f"rank={rank}",
                *( [f"g={self.coupling:g}"] if self.coupling is not None else [] ),
                f"v{self.scheme}",
            ]
        )

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        for k, v in d.items():
            if isinstance(v, Enum):
                d[k] = v.value
        return d

    def __str__(self) -> str:
        return self.slug()
