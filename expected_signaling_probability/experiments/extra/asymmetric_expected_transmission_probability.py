from expected_signaling_probability.utils.plotting import (
    apply_plot_style,
    LatexStrings,
    plot_scatter,
    plot_error_bars,
    plot_power_law_fit,
)
from expected_signaling_probability.utils.math import expected_transmission_probability, Direction
from expected_signaling_probability.utils.stats import statistics, Stats
import matplotlib.pyplot as plt
import numpy as np


def compute_asymmetric_expected_transmission_probability(
    n_samples: int, d_A_min: int, d_A_max: int, d_B: int, direction: Direction,
) -> list[Stats]:
    assert d_A_min <= d_A_max
    all_stats = []
    for d_A in range(d_A_min, d_A_max + 1):
        samples = expected_transmission_probability(n_samples, d_A, d_B, direction)
        stat = statistics(samples, d_A=d_A, d_B=d_B, direction=direction)
        all_stats.append(stat)
    return all_stats


DIRECTION_STYLE: dict[Direction, tuple[str, str]] = {
    Direction.A_TO_B: (LatexStrings.EXPECTED_TRANSMISSION_PROBABILITY_A_TO_B, "blue"),
    Direction.B_TO_A: (LatexStrings.EXPECTED_TRANSMISSION_PROBABILITY_B_TO_A, "red"),
}


def plot_asymmetric_expected_transmission_probability(
    all_stats_A_to_B: list[Stats],
    all_stats_B_to_A: list[Stats],
    use_error_bars: bool = True,
    d_fit_min: int | None = None,
):
    apply_plot_style()
    plt.figure(figsize=(10, 6))

    for all_stats, direction in [
        (all_stats_A_to_B, Direction.A_TO_B),
        (all_stats_B_to_A, Direction.B_TO_A),
    ]:
        label, color = DIRECTION_STYLE[direction]
        x = np.array([s.d_A for s in all_stats])
        y = np.array([s.mean for s in all_stats])

        plot_scatter(x, y, color=color, label=label)
        plot_power_law_fit(x, y, color=color, probability_label=label, dim_var="d_A", d_fit_min=d_fit_min)
        if use_error_bars:
            plot_error_bars(x, y, all_stats)

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel(r"$d_A$")
    plt.ylabel(LatexStrings.EXPECTED_TRANSMISSION_PROBABILITY_X_TO_Y)
    plt.title(f"Asymmetric Expected Transmission Probability ($N =$ {LatexStrings.n_samples_to_sci(all_stats_A_to_B[0].n)})")
    plt.legend()
    plt.tight_layout()
    plt.show()


def main():
    n_samples = 1_000
    d_A_min = 2
    d_A_max = 20
    d_B = 2
    d_fit_min = d_A_max // 2
    all_stats_A_to_B = compute_asymmetric_expected_transmission_probability(n_samples, d_A_min, d_A_max, d_B, Direction.A_TO_B)
    all_stats_B_to_A = compute_asymmetric_expected_transmission_probability(n_samples, d_A_min, d_A_max, d_B, Direction.B_TO_A)

    plot_asymmetric_expected_transmission_probability(all_stats_A_to_B, all_stats_B_to_A, d_fit_min=d_fit_min)


if __name__ == "__main__":
    main()
