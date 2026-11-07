"""Robustness of the headline (symmetric, unitary global op) to the other two ensemble choices.

Each curve changes exactly one factor relative to the headline:
  * state  Haar pure  -> Hilbert-Schmidt mixed
  * state  Haar pure  -> product of Hilbert-Schmidt states
  * local  Haar unitary -> BCSZ channel
"""
from expected_signaling_probability import Direction, DrawConfig, GlobalOpEnsemble, LocalOpEnsemble, StateEnsemble, sweep
from expected_signaling_probability.experiments.common import (
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
HEADLINE_GLOBAL_OP = GlobalOpEnsemble.HAAR_UNITARY

HEADLINE = DrawConfig(d_A=2, d_B=2, direction=Direction.A_TO_B, global_op=HEADLINE_GLOBAL_OP)

VARIANTS: list[tuple[str, str, DrawConfig]] = [
    ("headline (pure, unitary $\\mathcal{A}$)", "tab:blue", HEADLINE),
    ("HS mixed state", "tab:purple", HEADLINE.with_(state=StateEnsemble.HS_MIXED)),
    ("product HS state", "tab:olive", HEADLINE.with_(state=StateEnsemble.PRODUCT_HS)),
    ("BCSZ local op", "tab:brown", HEADLINE.with_(local_op=LocalOpEnsemble.BCSZ)),
]


def main(n_samples: int = N_SAMPLES, mode: PlotMode = PlotMode.PAPER):
    curves = [
        Curve(label=label, color=color, samples=sweep(symmetric_configs(cfg, D_MIN, D_MAX), n_samples, MASTER_SEED))
        for label, color, cfg in VARIANTS
    ]
    print_summary(curves)
    plot_figure(
        Figure(
            name="robustness_symmetric",
            curves=curves,
            xlabel=r"$d$",
            ylabel=LatexStrings.SYMMETRIC_EXPECTED_SIGNALING_PROBABILITY,
            dim_var="d",
            title=rf"Robustness to ensemble choice ($N = {n_samples}$)",
            d_fit_min=D_FIT_MIN,
            mode=mode,
        )
    )


if __name__ == "__main__":
    main()
