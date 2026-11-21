"""Figure 1: mean S and T against d = d_A = d_B for the three global ensembles.

S sits on the T curve in every ensemble: the signalling probability decays
exactly as fast as the transmission does, and in the unitary case neither decays.
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

D_MIN = 2
D_MAX = {GlobalOpEnsemble.HAAR_UNITARY: 10, GlobalOpEnsemble.COMS: 10, GlobalOpEnsemble.BCSZ: 8}
D_FIT_MIN = 5
BASE = DrawConfig(d_A=2, d_B=2, direction=Direction.A_TO_B)  # Haar pure state, Haar unitary local op


def curves(n_samples: int = N_SAMPLES, global_ops=(GlobalOpEnsemble.BCSZ,)) -> list[Curve]:
    out: list[Curve] = []
    for k, g in enumerate(global_ops):
        cfgs = symmetric_configs(BASE.with_(global_op=g), D_MIN, D_MAX[g])
        s = sweep(cfgs, n_samples, MASTER_SEED)
        t = sweep([c.with_(quantity=Quantity.TRANSMISSION) for c in cfgs], n_samples, MASTER_SEED)
        s_label = r"$\langle\mathcal{S}\rangle$" + (f" {GLOBAL_OP_LABEL[g]}" if len(global_ops) > 1 else "")
        out.append(Curve(label=s_label, color=GLOBAL_OP_COLOR[g], samples=s))
        t_label = (r"$\langle\mathcal{T}\rangle$" + (" (dotted)" if len(global_ops) > 1 else "")) if k == 0 else "_nolegend_"
        out.append(Curve(label=t_label, color=GLOBAL_OP_COLOR[g], samples=t, marker="s", linestyle=":", fit=False))
    return out


def main(n_samples: int = N_SAMPLES, mode: PlotMode = PlotMode.PAPER, global_ops=(GlobalOpEnsemble.BCSZ,), suffix: str = ""):
    cs = curves(n_samples, global_ops)
    print_summary(cs)
    plot_figure(
        Figure(
            name=f"fig_symmetric{suffix}",
            curves=cs,
            xlabel=r"$d$",
            ylabel=LatexStrings.SYMMETRIC_EXPECTED_SIGNALING_PROBABILITY + ", " + LatexStrings.SYMMETRIC_EXPECTED_TRANSMISSION_PROBABILITY,
            dim_var="d",
            title=rf"Symmetric $d_A = d_B = d$, mean with SEM ($N = {n_samples}$)",
            d_fit_min=D_FIT_MIN,
            mode=mode,
        )
    )


if __name__ == "__main__":
    main()
