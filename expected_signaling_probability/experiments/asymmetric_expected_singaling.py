from expected_signaling_probability.utils.plotting import apply_plot_style, LatexStrings
from expected_signaling_probability import expected_signaling_probability, Direction
from expected_signaling_probability.utils.stats import statistics, Stats
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np


def compute_asymmetric_expected_signaling_probability(n_samples: int, d_A_min: int, d_A_max: int, d_B: int, direction: Direction) -> list[Stats]:
    assert d_A_min <= d_A_max
    dims_A = [d for d in range(d_A_min, d_A_max + 1)]
    all_stats = []
    for d_A in dims_A:
        samples = expected_signaling_probability(n_samples, d_A, d_B, direction)
        stats = statistics(samples, d_A=d_A, d_B=d_B, direction=direction)
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


def add_plot_power_law_fit(x: np.typing.NDArray, y: np.typing.NDArray, direction: Direction):
    x_fit, y_fit, slope, _, r2, stderr, *_ = fit_power_law(x, y)
    plt.plot(
        x_fit,
        y_fit,
        label=rf"Fit: {LatexStrings.EXPECTED_SIGNALING_PROBABILITY_A_TO_B if direction == Direction.A_TO_B else LatexStrings.EXPECTED_SIGNALING_PROBABILITY_B_TO_A} $ \propto d_A^{{{slope:.2f} \pm {stderr:.2f}}}$",
        linestyle="--",
        color="blue" if direction == Direction.A_TO_B else "red",
    )


def add_plot_scatter(x: np.typing.NDArray, y: np.typing.NDArray, direction: Direction):
    plt.scatter(
        x,
        y,
        color="blue" if direction == Direction.A_TO_B else "red",
        s=50,
        label=rf"Data: {LatexStrings.EXPECTED_SIGNALING_PROBABILITY_A_TO_B if direction == Direction.A_TO_B else LatexStrings.EXPECTED_SIGNALING_PROBABILITY_B_TO_A}",
    )


def plot_asymmetric_expected_signaling_probability(
    all_stats_A_to_B: list[Stats],
    all_stats_B_to_A: list[Stats],
    use_error_bars: bool = True,
):
    apply_plot_style()
    x_atb = np.array([stat.d_A for stat in all_stats_A_to_B])
    y_atb = np.array([stat.mean for stat in all_stats_A_to_B])

    x_bta = np.array([stat.d_A for stat in all_stats_B_to_A])
    y_bta = np.array([stat.mean for stat in all_stats_B_to_A])
    plt.figure(figsize=(10, 6))

    add_plot_scatter(x_atb, y_atb, Direction.A_TO_B)
    add_plot_scatter(x_bta, y_bta, Direction.B_TO_A)

    add_plot_power_law_fit(x_atb, y_atb, Direction.A_TO_B)
    add_plot_power_law_fit(x_bta, y_bta, Direction.B_TO_A)

    if use_error_bars:
        add_plot_error_bars(x_atb, y_atb, all_stats_A_to_B)
        add_plot_error_bars(x_bta, y_bta, all_stats_B_to_A)

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel(r"$d_A$")
    plt.ylabel(LatexStrings.EXPECTED_SIGNALING_PROBABILITY_X_TO_Y)
    plt.title("Asymmetric Expected Signaling Probability" + " " + f"(N = {LatexStrings.n_samples_to_sci(all_stats_A_to_B[0].n)})")
    plt.legend()
    plt.tight_layout()
    plt.show()


def main():
    n_samples = 1000
    d_A_min = 2
    d_A_max = 20
    d_B = 2
    all_stats_A_to_B = compute_asymmetric_expected_signaling_probability(n_samples, d_A_min, d_A_max, d_B, Direction.A_TO_B)
    all_stats_B_to_A = compute_asymmetric_expected_signaling_probability(n_samples, d_A_min, d_A_max, d_B, Direction.B_TO_A)

    plot_asymmetric_expected_signaling_probability(all_stats_A_to_B, all_stats_B_to_A)


if __name__ == "__main__":
    main()
