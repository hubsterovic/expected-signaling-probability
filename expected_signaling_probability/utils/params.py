from dataclasses import dataclass


@dataclass
class ExtraParams:
    superoperator_rank: int | None = None


DEFAULT_EXTRA_PARAMS = ExtraParams()
