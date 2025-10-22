from expected_signaling_probability.utils.plotting import apply_plot_style, LatexStrings
from expected_signaling_probability import expected_signaling_probability, Direction
from expected_signaling_probability.utils.stats import statistics, Stats
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from scipy import stats

def compute_symmetric_expected_signaling_probability(n_samples: int, d_min: int, d_max: int) -> list[Stats]:
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

    lin = stats.linregress(log_x, log_y)
    r2 = float(lin.rvalue**2)  # type: ignore
    stderr = float(lin.stderr)  # type: ignore
    pval = float(lin.pvalue)  # type: ignore
    prefactor = float(10**intercept)
    return x_fit, y_fit, slope, intercept, r2, stderr, pval, prefactor


def add_plot_error_bars(x: np.typing.NDArray, y: np.typing.NDArray, all_stats: list[Stats]):
    yerr_low = y - np.array([stat.q25 for stat in all_stats])
    yerr_high = np.array([stat.q75 for stat in all_stats]) - y
    plt.errorbar(
        x,
        y,
        yerr=[yerr_low, yerr_high],
        label=r"IQR: $\pm 25\%$",
        fmt="none",
        capsize=3,
        ecolor="grey",
    )


def add_plot_power_law_fit(x: np.typing.NDArray, y: np.typing.NDArray):
    x_fit, y_fit, slope, _, r2, stderr, *_ = fit_power_law(x, y)
    plt.plot(
        x_fit,
        y_fit,
        label=rf"Fit: {LatexStrings.SYMMETRIC_EXPECTED_SIGNALING_PROBABILITY} $ \propto d^{{{slope:.2f} \pm {stderr:.2f}}}$",
        linestyle="--",
        color="purple",
    )


def add_plot_scatter(x: np.typing.NDArray, y: np.typing.NDArray):
    plt.scatter(
        x,
        y,
        color="purple",
        s=50,
        label=rf"Data: {LatexStrings.SYMMETRIC_EXPECTED_SIGNALING_PROBABILITY}",
    )


def plot_symmetric_expected_signaling_probability(all_stats: list[Stats], use_error_bars: bool = True, save: bool = True):
    apply_plot_style()
    x = np.array([stat.d_A for stat in all_stats])
    y = np.array([stat.mean for stat in all_stats])
    plt.figure(figsize=(10, 6))

    add_plot_scatter(x, y)
    add_plot_power_law_fit(x, y)
    if use_error_bars:
        add_plot_error_bars(x, y, all_stats)

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel(r"$d$")
    plt.xticks(x, [str(int(d)) for d in x])
    plt.ylabel(LatexStrings.SYMMETRIC_EXPECTED_SIGNALING_PROBABILITY)
    plt.title("Symmetric Expected Signaling Probability" + " " + LatexStrings.SYMMETRIC_EXPECTED_SIGNALING_PROBABILITY + " " + rf"($N =$ {LatexStrings.n_samples_to_sci(all_stats[0].n)})")
    plt.legend()
    plt.tight_layout()

    if save:
        filename = f"plots/symmetric_expected_signaling_probability_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.show()


def main():
    n_samples = 1000
    d_min = 2
    d_max = 10
    all_stats = compute_symmetric_expected_signaling_probability(n_samples, d_min, d_max)
    plot_symmetric_expected_signaling_probability(all_stats)


if __name__ == "__main__":
    main()
