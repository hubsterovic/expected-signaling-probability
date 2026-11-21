"""Figure 3: S and T against d_A at fixed d_B = 2, both directions, one panel per global ensemble.

Fitted exponents are in plots/exponents.md. The grey guide lines are the Page rule S, T ~ sqrt(d_receiver / d_discarded) with
d_discarded = d_sender * K, K the Kraus rank (1, D, D^2), anchored at the largest d_A.
"""
import matplotlib.pyplot as plt
import numpy as np

from expected_signaling_probability import Direction, DrawConfig, GlobalOpEnsemble, Quantity, sweep
from expected_signaling_probability.experiments.common import (
    GLOBAL_OP_LABEL,
    MASTER_SEED,
    N_SAMPLES,
    Curve,
    ErrorBar,
    PlotMode,
    asymmetric_configs,
    draw_curves,
    page_exponent,
    print_summary,
)
from expected_signaling_probability.utils.plotting import LatexStrings, apply_plot_style, plot_title, save_plot

D_B = 2
D_A_MIN = 2
D_A_MAX = {GlobalOpEnsemble.HAAR_UNITARY: 32, GlobalOpEnsemble.COMS: 32, GlobalOpEnsemble.BCSZ: 16}
D_FIT_MIN = {GlobalOpEnsemble.HAAR_UNITARY: 16, GlobalOpEnsemble.COMS: 16, GlobalOpEnsemble.BCSZ: 8}
BASE = DrawConfig(d_A=2, d_B=D_B)  # Haar pure state, Haar unitary local op

DIRECTION_COLOR = {Direction.A_TO_B: "tab:blue", Direction.B_TO_A: "tab:red"}
S_LABEL = {Direction.A_TO_B: LatexStrings.EXPECTED_SIGNALING_PROBABILITY_A_TO_B, Direction.B_TO_A: LatexStrings.EXPECTED_SIGNALING_PROBABILITY_B_TO_A}
T_LABEL = {Direction.A_TO_B: LatexStrings.EXPECTED_TRANSMISSION_PROBABILITY_A_TO_B, Direction.B_TO_A: LatexStrings.EXPECTED_TRANSMISSION_PROBABILITY_B_TO_A}


def panel_curves(g: GlobalOpEnsemble, n_samples: int = N_SAMPLES) -> list[Curve]:
    out: list[Curve] = []
    for direction in Direction:
        cfgs = asymmetric_configs(BASE.with_(global_op=g, direction=direction), D_A_MIN, D_A_MAX[g], D_B)
        s = sweep(cfgs, n_samples, MASTER_SEED)
        t = sweep([c.with_(quantity=Quantity.TRANSMISSION) for c in cfgs], n_samples, MASTER_SEED)
        out.append(Curve(label=S_LABEL[direction], color=DIRECTION_COLOR[direction], samples=s))
        out.append(Curve(label=T_LABEL[direction], color=DIRECTION_COLOR[direction], samples=t, marker="s", linestyle=":", fit=False))
    return out


def draw_page_guides(g: GlobalOpEnsemble, curves: list[Curve]) -> None:
    labelled = False
    for c in curves:
        if c.fit:  # one guide per S curve
            direction = c.samples[0].config.direction
            power = page_exponent(g, direction, D_B)
            if power is None:
                continue
            x = np.array([D_FIT_MIN[g], D_A_MAX[g]], dtype=float)
            y = c.y[-1] * (x / x[-1]) ** power
            plt.plot(x, y, color="0.4", linewidth=0.8, alpha=0.7, label="_nolegend_" if labelled else "Page rule")
            labelled = True


def main(n_samples: int = N_SAMPLES, mode: PlotMode = PlotMode.PAPER, global_ops=(GlobalOpEnsemble.BCSZ,), suffix: str = ""):
    apply_plot_style(mode)
    w, h = plt.rcParams["figure.figsize"]
    fig, axes = plt.subplots(1, len(global_ops), figsize=(w * len(global_ops), h), sharey=False)
    for ax, g in zip(np.atleast_1d(axes), global_ops):
        plt.sca(ax)
        cs = panel_curves(g, n_samples)
        print_summary(cs)
        draw_curves(cs, error_bar=ErrorBar.SEM, dim_var="d_A", d_fit_min=D_FIT_MIN[g], log_y=True, xlabel=r"$d_A$", ylabel="", exponent_in_label=False)
        draw_page_guides(g, cs)
        if len(global_ops) > 1:
            ax.set_title(GLOBAL_OP_LABEL[g])
        ax.set_xticks([t for t in (2, 4, 8, 16, 32) if t <= D_A_MAX[g]])
        ax.legend(loc="lower left")
    np.atleast_1d(axes)[0].set_ylabel(LatexStrings.EXPECTED_SIGNALING_PROBABILITY_X_TO_Y + ", " + LatexStrings.EXPECTED_TRANSMISSION_PROBABILITY_X_TO_Y)
    plot_title(rf"Asymmetric, $d_B = {D_B}$, mean with SEM ($N = {n_samples}$)")
    plt.tight_layout()
    save_plot(f"fig_asymmetric{suffix}", timestamped=mode is PlotMode.EXPLORE)
    plt.close(fig)


if __name__ == "__main__":
    main()
