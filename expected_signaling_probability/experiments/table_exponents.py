"""Table: fitted power-law exponents of S and T against the Page prediction.

Asymmetric regime d_B = 2: predicted exponent of d_A is (p_receiver - p_sender - p_K) / 2
with p_X the power of d_A in dim(X) and K the Kraus rank. Writes plots/exponents.md and .tex.
"""
from __future__ import annotations

import pandas as pd

from expected_signaling_probability import Direction, Quantity
from expected_signaling_probability.experiments import fig_asymmetric as fa
from expected_signaling_probability.experiments.common import GLOBAL_OP_LABEL, PAPER_GLOBAL_OPS, Curve, kraus_rank, page_exponent
from expected_signaling_probability.utils.fitting import fit_power_law
from expected_signaling_probability.utils.plotting import PLOTS_DIR

K_LABEL = {1: "1", None: "D^2"}


def fitted(c: Curve, d_fit_min: int) -> tuple[float, float]:
    mask = c.x >= d_fit_min
    fit = fit_power_law(c.x[mask], c.y[mask])
    return fit.slope, fit.stderr


def _markdown(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines) + "\n"


def _latex(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    body = [" & ".join(str(v).replace("±", r"\pm").replace("->", r"$\to$").replace(">=", r"$\geq$") for v in row) + r" \\" for row in df.itertuples(index=False)]
    return "\n".join([r"\begin{tabular}{" + "l" * len(cols) + "}", r"\toprule", " & ".join(cols) + r" \\", r"\midrule", *body, r"\bottomrule", r"\end{tabular}", ""])


def main() -> pd.DataFrame:
    rows = []
    for g in PAPER_GLOBAL_OPS:
        curves = fa.panel_curves(g)
        for c in curves:
            cfg = c.samples[0].config
            slope, err = fitted(c, fa.D_FIT_MIN[g])
            pred = page_exponent(g, cfg.direction, fa.D_B)
            rows.append(
                {
                    "ensemble": GLOBAL_OP_LABEL[g],
                    "K": {1: "1"}.get(kraus_rank(g, 2, 2), "D" if g.value == "coms" else "D^2"),
                    "direction": "A->B" if cfg.direction is Direction.A_TO_B else "B->A",
                    "quantity": "S" if cfg.quantity is Quantity.SIGNALING else "T",
                    "fit range": f"d_A >= {fa.D_FIT_MIN[g]}",
                    "predicted": "saturates" if pred is None else f"{pred:+.1f}",
                    "fitted": f"{slope:+.2f} ± {err:.2f}",
                }
            )
    df = pd.DataFrame(rows)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    (PLOTS_DIR / "exponents.md").write_text(_markdown(df))
    (PLOTS_DIR / "exponents.tex").write_text(_latex(df))
    print(df.to_string(index=False))
    return df


if __name__ == "__main__":
    main()
