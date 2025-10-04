from expected_signaling_probability.utils.directions import Direction
from dataclasses import dataclass
import numpy as np


@dataclass
class Stats:
    mean: np.floating
    median: np.floating
    q25: np.floating
    q75: np.floating
    std: np.floating
    var: np.floating
    min: np.floating
    max: np.floating
    n: int
    samples: np.typing.NDArray
    d_A: int
    d_B: int
    direction: Direction

    def __str__(self) -> str:
        attrs = []
        for field in self.__dataclass_fields__:
            value = getattr(self, field)
            if field == "samples":
                value = f"array with shape {self.samples.shape}"
            attrs.append(f"{field}={value}")
        return f"Stats({', '.join(attrs)})"


def statistics(samples: np.typing.NDArray, **kwargs) -> Stats:
    return Stats(
        mean=np.mean(samples),
        median=np.median(samples),
        q25=np.percentile(samples, 25),
        q75=np.percentile(samples, 75),
        std=np.std(samples),
        var=np.var(samples),
        min=np.min(samples),
        max=np.max(samples),
        n=len(samples),
        samples=samples,
        **kwargs,
    )
