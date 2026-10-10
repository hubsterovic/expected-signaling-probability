from expected_signaling_probability.utils.plotting import (
    PlotMode,
    apply_plot_style,
    plot_title,
    format_log_ticks,
    LatexStrings,
    plot_error_bars,
    plot_power_law_fit,
    save_plot,
)
from expected_signaling_probability import expected_signaling_probability, Direction, ExtraParams
from expected_signaling_probability.utils.stats import statistics, Stats
import matplotlib.pyplot as plt
import numpy as np


def compute(
    n_samples: int, d_A_min: int, d_A_max: int, d_B: int,
    direction: Direction, superoperator_rank: int | None,
) -> list[Stats]:
    assert d_A_min <= d_A_max
    extra_params = ExtraParams(superoperator_rank=superoperator_rank)
    all_stats = []
    for d_A in range(d_A_min, d_A_max + 1):
        samples = expected_signaling_probability(n_samples, d_A, d_B, direction, extra_params=extra_params)
        stat = statistics(samples, d_A=d_A, d_B=d_B, direction=direction)
        all_stats.append(stat)
    return all_stats


STYLE_MAP: dict[tuple[Direction, str], tuple[str, str, str]] = {
    (Direction.A_TO_B, "full"): (r"$\langle \mathcal{S} \rangle_{A \to B}$ (CPTP)", "blue", "o"),
    (Direction.A_TO_B, "unitary"): (r"$\langle \mathcal{S} \rangle_{A \to B}$ (Unitary)", "blue", "^"),
    (Direction.B_TO_A, "full"): (r"$\langle \mathcal{S} \rangle_{B \to A}$ (CPTP)", "red", "o"),
    (Direction.B_TO_A, "unitary"): (r"$\langle \mathcal{S} \rangle_{B \to A}$ (Unitary)", "red", "^"),
}


def plot_unitary_comparison(
    results: dict[tuple[Direction, str], list[Stats]],
    d_B: int,
    use_error_bars: bool = True,
    save: bool = True,
    d_fit_min: int | None = None,
    mode: PlotMode = PlotMode.EXPLORE,
):
    apply_plot_style(mode)
    plt.figure()

    for (direction, rank_label), all_stats in results.items():
        label, color, marker = STYLE_MAP[(direction, rank_label)]
        x = np.array([s.d_A for s in all_stats])
        y = np.array([s.mean for s in all_stats])

        fill_kwargs = {"facecolors": "none", "edgecolors": color, "linewidths": 0.8}
        plt.scatter(x, y, s=12, zorder=2, marker=marker, label=label, **fill_kwargs)

        plot_power_law_fit(x, y, color=color, probability_label=label, dim_var="d_A", d_fit_min=d_fit_min)

        if use_error_bars:
            plot_error_bars(x, y, all_stats)

    plt.xscale("log")
    plt.yscale("log")
    format_log_ticks()
    plt.xlabel(r"$d_A$")
    plt.ylabel(LatexStrings.EXPECTED_SIGNALING_PROBABILITY_X_TO_Y)
    plot_title(rf"Unitary vs CPTP ($d_B =$ {d_B}, $N =$ {LatexStrings.n_samples_to_sci(list(results.values())[0][0].n)})")
    plt.legend()
    plt.tight_layout()

    if save:
        save_plot("asymmetric_unitary_comparison")
    plt.show()


def main():
    n_samples = 1_000
    d_A_min = 2
    d_A_max = 16
    d_B = 2
    d_fit_min = d_A_max // 2
    plot_mode = PlotMode.PAPER

    results: dict[tuple[Direction, str], list[Stats]] = {}

    for rank_label, rank_value in [("full", None), ("unitary", 1)]:
        for direction in [Direction.A_TO_B, Direction.B_TO_A]:
            results[(direction, rank_label)] = compute(
                n_samples, d_A_min, d_A_max, d_B, direction, rank_value,
            )

    plot_unitary_comparison(results, d_B=d_B, d_fit_min=d_fit_min, mode=plot_mode)


if __name__ == "__main__":
    main()
