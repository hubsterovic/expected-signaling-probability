"""Headline, symmetric regime d_A = d_B = d.

State: Haar pure. Local op: Haar unitary. Global op: one curve each for
unitary, COMS and full-rank BCSZ. The BCSZ transmission baseline T is
drawn alongside so that erasure and causality can be told apart.
"""
from expected_signaling_probability import Direction, DrawConfig, GlobalOpEnsemble, Quantity, sweep
from expected_signaling_probability.experiments.common import (
    GLOBAL_OP_COLOR,
    GLOBAL_OP_LABEL,
    MASTER_SEED,
    N_SAMPLES,
    Curve,
    Figure,
    PlotMode,
    plot_figure,
    print_summary,
    symmetric_configs,
)
from expected_signaling_probability.utils.plotting import LatexStrings

D_MIN, D_MAX = 2, 10
D_FIT_MIN = 5

BASE = DrawConfig(d_A=2, d_B=2, direction=Direction.A_TO_B)  # Haar pure state, Haar unitary local op (defaults)


def main(n_samples: int = N_SAMPLES, mode: PlotMode = PlotMode.PAPER):
    curves: list[Curve] = []
    for g in GlobalOpEnsemble:
        cfgs = symmetric_configs(BASE.with_(global_op=g), D_MIN, D_MAX)
        label = rf"{LatexStrings.SYMMETRIC_EXPECTED_SIGNALING_PROBABILITY} ({GLOBAL_OP_LABEL[g]})"
        curves.append(Curve(label=label, color=GLOBAL_OP_COLOR[g], samples=sweep(cfgs, n_samples, MASTER_SEED)))

    baseline_cfgs = symmetric_configs(BASE.with_(global_op=GlobalOpEnsemble.BCSZ, quantity=Quantity.TRANSMISSION), D_MIN, D_MAX)
    curves.append(
        Curve(
            label=rf"{LatexStrings.SYMMETRIC_EXPECTED_TRANSMISSION_PROBABILITY} (BCSZ)",
            color="tab:gray",
            marker="s",
            samples=sweep(baseline_cfgs, n_samples, MASTER_SEED),
        )
    )

    print_summary(curves)
    plot_figure(
        Figure(
            name="headline_symmetric",
            curves=curves,
            xlabel=r"$d$",
            ylabel=LatexStrings.SYMMETRIC_EXPECTED_SIGNALING_PROBABILITY,
            dim_var="d",
            title=rf"Symmetric expected signalling probability ($N = {n_samples}$)",
            d_fit_min=D_FIT_MIN,
            mode=mode,
        )
    )


if __name__ == "__main__":
    main()
