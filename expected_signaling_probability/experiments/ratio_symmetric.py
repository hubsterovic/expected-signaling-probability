"""Pointwise normalized signalling S/T, symmetric regime, median with IQR.

1 - med(S/T) is the causal suppression beyond erasure; it should vanish with d.
"""
from expected_signaling_probability import Direction, DrawConfig, GlobalOpEnsemble, Quantity, sweep
from expected_signaling_probability.experiments.common import (
    GLOBAL_OP_COLOR,
    GLOBAL_OP_LABEL,
    MASTER_SEED,
    Curve,
    ErrorBar,
    Figure,
    PlotMode,
    plot_figure,
    print_summary,
    symmetric_configs,
)
from expected_signaling_probability.utils.plotting import LatexStrings

N_SAMPLES = 2_000
D_MIN, D_MAX = 2, 10
BASE = DrawConfig(d_A=2, d_B=2, direction=Direction.A_TO_B, quantity=Quantity.RATIO)


def main(n_samples: int = N_SAMPLES, mode: PlotMode = PlotMode.PAPER, global_ops=tuple(GlobalOpEnsemble)):
    curves = [
        Curve(
            label=rf"{LatexStrings.MEDIAN_RATIO} ({GLOBAL_OP_LABEL[g]})",
            color=GLOBAL_OP_COLOR[g],
            samples=sweep(symmetric_configs(BASE.with_(global_op=g), D_MIN, D_MAX), n_samples, MASTER_SEED),
            y_stat="median",
            fit=False,
        )
        for g in global_ops
    ]
    print_summary(curves)
    plot_figure(
        Figure(
            name="ratio_symmetric",
            curves=curves,
            xlabel=r"$d$",
            ylabel=LatexStrings.MEDIAN_RATIO,
            title=rf"Normalized signalling, median with IQR ($N = {n_samples}$)",
            error_bar=ErrorBar.IQR,
            log_y=False,
            mode=mode,
        )
    )


if __name__ == "__main__":
    main()
