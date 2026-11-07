"""Headline, asymmetric regime d_B = 2 fixed, d_A swept, both directions.

One figure per global ensemble, with A->B and B->A on the same axes. The BCSZ
figure also shows the transmission baseline T in both directions.
"""
from expected_signaling_probability import Direction, DrawConfig, GlobalOpEnsemble, Quantity, sweep
from expected_signaling_probability.experiments.common import (
    GLOBAL_OP_LABEL,
    MASTER_SEED,
    N_SAMPLES,
    Curve,
    Figure,
    PlotMode,
    asymmetric_configs,
    plot_figure,
    print_summary,
)
from expected_signaling_probability.utils.plotting import LatexStrings

D_A_MIN, D_A_MAX = 2, 32
D_B = 2
D_FIT_MIN = 16

BASE = DrawConfig(d_A=2, d_B=D_B)  # Haar pure state, Haar unitary local op (defaults)

DIRECTION_STYLE = {
    Direction.A_TO_B: (LatexStrings.EXPECTED_SIGNALING_PROBABILITY_A_TO_B, "tab:blue"),
    Direction.B_TO_A: (LatexStrings.EXPECTED_SIGNALING_PROBABILITY_B_TO_A, "tab:red"),
}
BASELINE_STYLE = {
    Direction.A_TO_B: (LatexStrings.EXPECTED_TRANSMISSION_PROBABILITY_A_TO_B, "tab:cyan"),
    Direction.B_TO_A: (LatexStrings.EXPECTED_TRANSMISSION_PROBABILITY_B_TO_A, "tab:orange"),
}


def main(n_samples: int = N_SAMPLES, mode: PlotMode = PlotMode.PAPER, global_ops: tuple[GlobalOpEnsemble, ...] = tuple(GlobalOpEnsemble)):
    for g in global_ops:
        curves: list[Curve] = []
        for direction, (label, color) in DIRECTION_STYLE.items():
            cfgs = asymmetric_configs(BASE.with_(global_op=g, direction=direction), D_A_MIN, D_A_MAX, D_B)
            curves.append(Curve(label=label, color=color, samples=sweep(cfgs, n_samples, MASTER_SEED)))
        if g is GlobalOpEnsemble.BCSZ:
            for direction, (label, color) in BASELINE_STYLE.items():
                cfgs = asymmetric_configs(BASE.with_(global_op=g, direction=direction, quantity=Quantity.TRANSMISSION), D_A_MIN, D_A_MAX, D_B)
                curves.append(Curve(label=label, color=color, marker="s", samples=sweep(cfgs, n_samples, MASTER_SEED)))

        print_summary(curves)
        plot_figure(
            Figure(
                name=f"headline_asymmetric_{g.value}",
                curves=curves,
                xlabel=r"$d_A$",
                ylabel=LatexStrings.EXPECTED_SIGNALING_PROBABILITY_X_TO_Y,
                dim_var="d_A",
                title=rf"Asymmetric expected signalling probability, {GLOBAL_OP_LABEL[g]} ($d_B = {D_B}$, $N = {n_samples}$)",
                d_fit_min=D_FIT_MIN,
                mode=mode,
            )
        )


if __name__ == "__main__":
    main()
