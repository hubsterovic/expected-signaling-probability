from expected_signaling_probability.utils.plotting import (
    apply_plot_style,
    LatexStrings,
    plot_scatter,
    plot_error_bars,
    plot_power_law_fit,
)
from expected_signaling_probability import expected_signaling_probability, Direction
from expected_signaling_probability.utils.stats import statistics, Stats
from datetime import datetime
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
):
    apply_plot_style()
    x = np.array([s.d_A for s in all_stats])
    y = np.array([s.mean for s in all_stats])
    plt.figure(figsize=(10, 6))

    label = LatexStrings.SYMMETRIC_EXPECTED_SIGNALING_PROBABILITY
    plot_scatter(x, y, color="purple", label=label)
    plot_power_law_fit(x, y, color="purple", probability_label=label, dim_var="d", d_fit_min=d_fit_min)
    if use_error_bars:
        plot_error_bars(x, y, all_stats)

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel(r"$d$")
    plt.xticks(x, [str(int(d)) for d in x])
    plt.ylabel(label)
    plt.title(f"Symmetric Expected Signaling Probability {label} ($N =$ {LatexStrings.n_samples_to_sci(all_stats[0].n)})")
    plt.legend()
    plt.tight_layout()

    if save:
        filename = f"plots/symmetric_expected_signaling_probability_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.show()


def main():
    n_samples = 1_000
    d_min = 2
    d_max = 10
    d_fit_min = d_max // 2
    all_stats = compute_symmetric_expected_signaling_probability(n_samples, d_min, d_max)
    plot_symmetric_expected_signaling_probability(all_stats, d_fit_min=d_fit_min)


if __name__ == "__main__":
    main()
