from dataclasses import dataclass
from enum import Enum


class StateEnsemble(str, Enum):
    GINIBRE = "ginibre"
    PURE = "pure"
    PRODUCT = "product"


@dataclass
class ExtraParams:
    superoperator_rank: int | None = None
    state_ensemble: StateEnsemble = StateEnsemble.GINIBRE


DEFAULT_EXTRA_PARAMS = ExtraParams()
