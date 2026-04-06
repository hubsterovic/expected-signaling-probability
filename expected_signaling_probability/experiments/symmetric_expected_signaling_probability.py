from expected_signaling_probability.utils.plotting import (
    PlotMode,
    apply_plot_style,
    plot_title,
    format_log_ticks,
    LatexStrings,
    plot_scatter,
    plot_error_bars,
    plot_power_law_fit,
    save_plot,
)
from expected_signaling_probability import expected_signaling_probability, Direction
from expected_signaling_probability.utils.stats import statistics, Stats
import matplotlib.pyplot as plt
import numpy as np


def compute_symmetric_expected_signaling_probability(n_samples: int, d_min: int, d_max: int) -> list[Stats]:
    assert d_min <= d_max
    all_stats = []
    for d in range(d_min, d_max + 1):
        samples = expected_signaling_probability(n_samples, d, d, Direction.A_TO_B)
        stat = statistics(samples, d_A=d, d_B=d, direction=Direction.A_TO_B)
        all_stats.append(stat)
    return all_stats


def plot_symmetric_expected_signaling_probability(
    all_stats: list[Stats],
    use_error_bars: bool = True,
    save: bool = True,
    d_fit_min: int | None = None,
    mode: PlotMode = PlotMode.EXPLORE,
):
    apply_plot_style(mode)
    x = np.array([s.d_A for s in all_stats])
    y = np.array([s.mean for s in all_stats])
    plt.figure()

    label = LatexStrings.SYMMETRIC_EXPECTED_SIGNALING_PROBABILITY
    plot_scatter(x, y, color="purple", label=label)
    plot_power_law_fit(x, y, color="purple", probability_label=label, dim_var="d", d_fit_min=d_fit_min)
    if use_error_bars:
        plot_error_bars(x, y, all_stats)

    plt.xscale("log")
    plt.yscale("log")
    format_log_ticks()
    plt.xlabel(r"$d$")
    plt.ylabel(label)
    plot_title(f"Symmetric Expected Signaling Probability {label} ($N =$ {LatexStrings.n_samples_to_sci(all_stats[0].n)})")
    plt.legend()
    plt.tight_layout()

    if save:
        save_plot("symmetric_expected_signaling_probability")
    plt.show()


def main():
    n_samples = 1_000
    d_min = 2
    d_max = 10
    all_stats = compute_symmetric_expected_signaling_probability(n_samples, d_min, d_max)

    plot_mode = PlotMode.PAPER
    d_fit_min = d_max // 2
    plot_symmetric_expected_signaling_probability(all_stats, d_fit_min=d_fit_min, mode=plot_mode)


if __name__ == "__main__":
    main()
