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
    SYMMETRIC_COLOR,
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
PRODUCT_GLOBAL_OPS = (GlobalOpEnsemble.HAAR_UNITARY, GlobalOpEnsemble.COMS, GlobalOpEnsemble.BCSZ)
D_MAX_PRODUCT = {GlobalOpEnsemble.HAAR_UNITARY: 10, GlobalOpEnsemble.COMS: 10, GlobalOpEnsemble.BCSZ: 8}
BASE = DrawConfig(d_A=2, d_B=2, direction=Direction.A_TO_B, quantity=Quantity.RATIO)
MEDIAN_RATIO_LOCAL = r"$\mathrm{med}\,(\mathcal{S}/\mathcal{L})$"


def haar_curves(global_ops=tuple(N_HAAR), label_ensemble: bool = True, color: str | None = None) -> list[Curve]:
    out: list[Curve] = []
    for g in global_ops:
        cfgs = symmetric_configs(BASE.with_(global_op=g), D_MIN, D_MAX[g])
        label = GLOBAL_OP_LABEL[g] if label_ensemble else "Haar inputs"
        out.append(Curve(label=label, color=color or GLOBAL_OP_COLOR[g], samples=sweep(cfgs, N_HAAR[g], MASTER_SEED), y_stat="median", fit=False))
    return out


def product_curves(global_ops=PRODUCT_GLOBAL_OPS, with_local: bool = True, label_ensemble: bool = True, color: str | None = None) -> list[Curve]:
    out: list[Curve] = []
    for g in global_ops:
        cfgs = symmetric_configs(BASE.with_(global_op=g, state=StateEnsemble.PRODUCT_HS), D_MIN, D_MAX_PRODUCT[g])
        name = f" {GLOBAL_OP_LABEL[g]}" if label_ensemble else ""
        label = rf"$\mathcal{{S}}/\mathcal{{T}}${name}" if with_local else "product inputs"
        out.append(Curve(label=label, color=color or GLOBAL_OP_COLOR[g], samples=sweep(cfgs, N_PRODUCT, MASTER_SEED), y_stat="median", fit=False, marker="^" if not with_local else "o", linestyle="-." if not with_local else "--"))
        if with_local and g is not GlobalOpEnsemble.BCSZ:
            cfgs_l = [c.with_(quantity=Quantity.RATIO_LOCAL) for c in cfgs]
            out.append(Curve(label=rf"$\mathcal{{S}}/\mathcal{{L}}${name}", color=color or GLOBAL_OP_COLOR[g], samples=sweep(cfgs_l, N_PRODUCT, MASTER_SEED), y_stat="median", fit=False, marker="^", linestyle="-."))
    return out


def main_bcsz(mode: PlotMode = PlotMode.PAPER):
    """Main-text figure: BCSZ only, Haar and product inputs on one axis."""
    apply_plot_style(mode)
    plt.figure()
    curves = haar_curves((GlobalOpEnsemble.BCSZ,), label_ensemble=False, color=SYMMETRIC_COLOR) + product_curves((GlobalOpEnsemble.BCSZ,), with_local=False, color=SYMMETRIC_COLOR)
    print_summary(curves)
    draw_curves(curves, error_bar=ErrorBar.IQR, dim_var="d", d_fit_min=None, log_y=False, xlabel=r"$d$", ylabel=LatexStrings.MEDIAN_RATIO)
    plt.axhline(1.0, color="0.5", linewidth=0.6, zorder=0)
    plt.ylim(0, 1.3)
    plot_title("Normalized signalling, BCSZ, median with IQR")
    plt.legend(loc="lower right")
    plt.tight_layout()
    save_plot("fig_ratio", timestamped=mode is PlotMode.EXPLORE)
    plt.close()


def main(mode: PlotMode = PlotMode.PAPER, suffix: str = "_all"):
    """Supplementary figure: all ensembles, Haar inputs (a) and product inputs with S/T and S/L (b)."""
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
    save_plot(f"fig_ratio{suffix}", timestamped=mode is PlotMode.EXPLORE)
    plt.close(fig)


if __name__ == "__main__":
    main_bcsz()
    main()
