import re
from expected_signaling_probability import expected_signaling_probability, Direction
from expected_signaling_probability.utils.stats import statistics, Stats
from expected_signaling_probability.utils.plotting import apply_plot_style
import matplotlib.pyplot as plt
import numpy as np


def compute_symmetric_expected_signaling_probability(
    n_samples: int, d_min: int, d_max: int
) -> list[Stats]:
    assert d_min <= d_max
    dims = [d for d in range(d_min, d_max + 1)]
    all_stats = []
    for d in dims:
        samples = expected_signaling_probability(n_samples, d, d, Direction.A_TO_B)
        stats = statistics(samples, d_A=d, d_B=d, direction=Direction.A_TO_B)
        all_stats.append(stats)
    return all_stats


def fit_power_law(x: np.typing.NDArray, y: np.typing.NDArray):
    log_x = np.log10(x)
    log_y = np.log10(np.maximum(y, np.finfo(float).tiny))
    slope, intercept = np.polyfit(log_x, log_y, 1)
    x_fit = np.logspace(np.log10(x.min()), np.log10(x.max()), 200)
    y_fit = 10 ** (intercept + slope * np.log10(x_fit))
    return x_fit, y_fit, slope, intercept



def plot_symmetric_expected_signaling_probability(all_stats: list[Stats]):
    apply_plot_style()
    x = np.array([stat.d_A if stat.d_A is not None else 0 for stat in all_stats])
    y = np.array([stat.mean for stat in all_stats])
    plt.figure(figsize=(10, 6))
    plt.scatter(x=x, y=y)
    x_fit, y_fit, slope, intercept = fit_power_law(x, y)
    plt.plot(x_fit, y_fit, label=f"Power Law Fit (slope={slope:.2f})")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Dimension")
    plt.ylabel("Expected Signaling Probability")
    plt.title("Symmetric Expected Signaling Probability")
    plt.legend()
    plt.show()

def main():
   n_samples = 10000
   d_min = 2
   d_max = 8
   all_stats = compute_symmetric_expected_signaling_probability(n_samples, d_min, d_max)
   plot_symmetric_expected_signaling_probability(all_stats)




if __name__ == "__main__":
    main()
