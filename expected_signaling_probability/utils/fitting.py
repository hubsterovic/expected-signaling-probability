from dataclasses import dataclass
from scipy import stats as sp_stats
import numpy as np


@dataclass
class PowerLawFit:
    x_fit: np.typing.NDArray
    y_fit: np.typing.NDArray
    slope: float
    intercept: float
    r2: float
    stderr: float
    pval: float
    prefactor: float


def fit_power_law(x: np.typing.NDArray, y: np.typing.NDArray) -> PowerLawFit:
    log_x = np.log10(x)
    log_y = np.log10(np.maximum(y, np.finfo(float).tiny))
    slope, intercept = np.polyfit(log_x, log_y, 1)
    x_fit = np.logspace(np.log10(x.min()), np.log10(x.max()), 200)
    y_fit = 10 ** (intercept + slope * np.log10(x_fit))

    lin = sp_stats.linregress(log_x, log_y)
    r2 = float(lin.rvalue**2)  # type: ignore
    stderr = float(lin.stderr)  # type: ignore
    pval = float(lin.pvalue)  # type: ignore
    prefactor = float(10**intercept)
    return PowerLawFit(
        x_fit=x_fit,
        y_fit=y_fit,
        slope=slope,
        intercept=intercept,
        r2=r2,
        stderr=stderr,
        pval=pval,
        prefactor=prefactor,
    )
