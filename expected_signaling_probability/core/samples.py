from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from expected_signaling_probability.core.config import DrawConfig


@dataclass(frozen=True)
class Samples:
    """The Monte Carlo draws for one configuration, with summary statistics."""

    config: DrawConfig
    master_seed: int
    values: np.ndarray

    @property
    def n(self) -> int:
        return int(self.values.size)

    @property
    def mean(self) -> float:
        return float(np.mean(self.values))

    @property
    def sem(self) -> float:
        """Standard error of the mean."""
        return float(np.std(self.values, ddof=1) / np.sqrt(self.n)) if self.n > 1 else float("nan")

    @property
    def std(self) -> float:
        return float(np.std(self.values, ddof=1)) if self.n > 1 else float("nan")

    @property
    def median(self) -> float:
        return float(np.median(self.values))

    @property
    def q25(self) -> float:
        return float(np.percentile(self.values, 25))

    @property
    def q75(self) -> float:
        return float(np.percentile(self.values, 75))

    def row(self) -> dict:
        return {
            **self.config.to_dict(),
            "master_seed": self.master_seed,
            "n": self.n,
            "mean": self.mean,
            "sem": self.sem,
            "std": self.std,
            "median": self.median,
            "q25": self.q25,
            "q75": self.q75,
        }

    def __str__(self) -> str:
        return f"{self.config.slug()}: n={self.n} mean={self.mean:.4g} ± {self.sem:.2g} (sem), median={self.median:.4g}, IQR=[{self.q25:.4g}, {self.q75:.4g}]"


def summary_table(samples: list[Samples]) -> pd.DataFrame:
    return pd.DataFrame([s.row() for s in samples])
