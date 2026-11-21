"""Figure 2: median of the pointwise normalized signalling against d = d_A = d_B, with IQR.

(a) Haar-random entangled inputs: med(S/T) for the three global ensembles.
(b) Product (Hilbert-Schmidt) inputs, unitary and COMS: med(S/T), which plateaus
    because T now also contains Bob's own-side change, and med(S/L), Alice's
    intervention against Bob's own local change on the same draw.
Neither ratio decays with d: there is no suppression of Alice beyond erasure.
"""
import matplotlib.pyplot as plt

from expected_signaling_probability import Direction, DrawConfig, GlobalOpEnsemble, Quantity, StateEnsemble, sweep
from expected_signaling_probability.experiments.common import (
    GLOBAL_OP_COLOR,
    GLOBAL_OP_LABEL,
    MASTER_SEED,
    Curve,
    ErrorBar,
    PlotMode,
    draw_curves,
    print_summary,
    symmetric_configs,
)
from expected_signaling_probability.utils.plotting import LatexStrings, apply_plot_style, plot_title, save_plot

D_MIN = 2
D_MAX = {GlobalOpEnsemble.HAAR_UNITARY: 10, GlobalOpEnsemble.COMS: 10, GlobalOpEnsemble.BCSZ: 9}
N_HAAR = {GlobalOpEnsemble.HAAR_UNITARY: 2_000, GlobalOpEnsemble.COMS: 2_000, GlobalOpEnsemble.BCSZ: 1_000}
N_PRODUCT = 1_000
PRODUCT_GLOBAL_OPS = (GlobalOpEnsemble.HAAR_UNITARY, GlobalOpEnsemble.COMS)
BASE = DrawConfig(d_A=2, d_B=2, direction=Direction.A_TO_B, quantity=Quantity.RATIO)
MEDIAN_RATIO_LOCAL = r"$\mathrm{med}\,(\mathcal{S}/\mathcal{L})$"


def haar_curves() -> list[Curve]:
    out: list[Curve] = []
    for g in N_HAAR:
        cfgs = symmetric_configs(BASE.with_(global_op=g), D_MIN, D_MAX[g])
        out.append(Curve(label=GLOBAL_OP_LABEL[g], color=GLOBAL_OP_COLOR[g], samples=sweep(cfgs, N_HAAR[g], MASTER_SEED), y_stat="median", fit=False))
    return out


def product_curves() -> list[Curve]:
    out: list[Curve] = []
    for g in PRODUCT_GLOBAL_OPS:
        cfgs = symmetric_configs(BASE.with_(global_op=g, state=StateEnsemble.PRODUCT_HS), D_MIN, D_MAX[g])
        out.append(Curve(label=rf"$\mathcal{{S}}/\mathcal{{T}}$ {GLOBAL_OP_LABEL[g]}", color=GLOBAL_OP_COLOR[g], samples=sweep(cfgs, N_PRODUCT, MASTER_SEED), y_stat="median", fit=False))
        cfgs_l = [c.with_(quantity=Quantity.RATIO_LOCAL) for c in cfgs]
        out.append(Curve(label=rf"$\mathcal{{S}}/\mathcal{{L}}$ {GLOBAL_OP_LABEL[g]}", color=GLOBAL_OP_COLOR[g], samples=sweep(cfgs_l, N_PRODUCT, MASTER_SEED), y_stat="median", fit=False, marker="^", linestyle="-."))
    return out


def main(mode: PlotMode = PlotMode.PAPER):
    apply_plot_style(mode)
    w, h = plt.rcParams["figure.figsize"]
    fig, axes = plt.subplots(1, 2, figsize=(2 * w, h), sharey=True)
    for ax, (title, curves) in zip(axes, [("Haar inputs", haar_curves()), ("product inputs", product_curves())]):
        plt.sca(ax)
        print_summary(curves)
        draw_curves(curves, error_bar=ErrorBar.IQR, dim_var="d", d_fit_min=None, log_y=False, xlabel=r"$d$", ylabel="")
        ax.axhline(1.0, color="0.5", linewidth=0.6, linestyle="-", zorder=0)
        ax.set_ylim(0, 1.3)
        ax.set_title(title)
        ax.legend(loc="lower right")
    axes[0].set_ylabel(LatexStrings.MEDIAN_RATIO + ", " + MEDIAN_RATIO_LOCAL)
    plot_title("Normalized signalling, median with IQR")
    plt.tight_layout()
    save_plot("fig_ratio", timestamped=mode is PlotMode.EXPLORE)
    plt.close(fig)


if __name__ == "__main__":
    main()
