from expected_signaling_probability.utils.fitting import fit_power_law
from expected_signaling_probability.utils.stats import Stats
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import shutil
import os


def apply_plot_style():
    sns.set_theme(
        style="whitegrid",
        context="paper",
        font_scale=1.5,
    )

    use_tex = shutil.which("pdflatex") is not None
    if use_tex:
        os.environ["PATH"] = "/Library/TeX/texbin:" + os.environ["PATH"]

    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "axes.linewidth": 1.2,
            "grid.linewidth": 0.7,
            "grid.alpha": 0.3,
            "axes.unicode_minus": False,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
        }
    )

    if use_tex:
        plt.rcParams.update(
            {
                "text.usetex": True,
                "font.family": "serif",
                "font.serif": ["Computer Modern Roman"],
                "text.latex.preamble": r"""
                    \usepackage{amsmath}
                    \usepackage{amssymb}
                """,
            }
        )
        print("[plot_styling] Using full LaTeX rendering via pdflatex.")
    else:
        plt.rcParams.update(
            {
                "text.usetex": False,
                "mathtext.fontset": "cm",
                "font.family": "serif",
            }
        )
        print("[plot_styling] LaTeX not found. Using mathtext fallback.")


class LatexStrings:
    EXPECTED_SIGNALING_PROBABILITY_X_TO_Y = r"$\langle \mathcal{S} \rangle_{X \to Y}$"
    EXPECTED_SIGNALING_PROBABILITY_A_TO_B = r"$\langle \mathcal{S} \rangle_{A \to B}$"
    EXPECTED_SIGNALING_PROBABILITY_B_TO_A = r"$\langle \mathcal{S} \rangle_{B \to A}$"
    SYMMETRIC_EXPECTED_SIGNALING_PROBABILITY = r"$\langle \mathcal{S} \rangle$"

    EXPECTED_TRANSMISSION_PROBABILITY_X_TO_Y = r"$\langle \mathcal{T} \rangle_{X \to Y}$"
    EXPECTED_TRANSMISSION_PROBABILITY_A_TO_B = r"$\langle \mathcal{T} \rangle_{A \to B}$"
    EXPECTED_TRANSMISSION_PROBABILITY_B_TO_A = r"$\langle \mathcal{T} \rangle_{B \to A}$"
    SYMMETRIC_EXPECTED_TRANSMISSION_PROBABILITY = r"$\langle \mathcal{T} \rangle$"

    EXPECTED_CORRELATION_PROBABILITY_X_TO_Y = r"$\langle \mathcal{C} \rangle_{X \to Y}$"
    EXPECTED_CORRELATION_PROBABILITY_A_TO_B = r"$\langle \mathcal{C} \rangle_{A \to B}$"
    EXPECTED_CORRELATION_PROBABILITY_B_TO_A = r"$\langle \mathcal{C} \rangle_{B \to A}$"
    SYMMETRIC_EXPECTED_CORRELATION_PROBABILITY = r"$\langle \mathcal{C} \rangle$"

    @classmethod
    def n_samples_to_sci(cls, n: int) -> str:
        p = np.log10(n)
        return f"$10^{{{p:.0f}}}$"


def plot_scatter(
    x: np.typing.NDArray,
    y: np.typing.NDArray,
    *,
    color: str,
    label: str,
):
    plt.scatter(x, y, color=color, s=50, label=rf"Data: {label}")


def plot_error_bars(
    x: np.typing.NDArray,
    y: np.typing.NDArray,
    all_stats: list[Stats],
):
    yerr_low = y - np.array([s.q25 for s in all_stats])
    yerr_high = np.array([s.q75 for s in all_stats]) - y
    plt.errorbar(
        x,
        y,
        yerr=[yerr_low, yerr_high],
        label=r"IQR: $\pm 25\%$",
        fmt="none",
        capsize=3,
        ecolor="grey",
    )


def plot_power_law_fit(
    x: np.typing.NDArray,
    y: np.typing.NDArray,
    *,
    color: str,
    probability_label: str,
    dim_var: str = "d",
    d_fit_min: int | None = None,
):
    if d_fit_min is not None:
        mask = x >= d_fit_min
        x_fit_data, y_fit_data = x[mask], y[mask]
    else:
        x_fit_data, y_fit_data = x, y

    fit = fit_power_law(x_fit_data, y_fit_data)
    fit_range = rf" (${dim_var} \geq {d_fit_min}$)" if d_fit_min is not None else ""
    plt.plot(
        fit.x_fit,
        fit.y_fit,
        label=rf"Fit{fit_range}: {probability_label} $ \propto {dim_var}^{{{fit.slope:.2f} \pm {fit.stderr:.2f}}}$",
        linestyle="--",
        color=color,
    )

    if d_fit_min is not None and x.min() < d_fit_min:
        x_extrap = np.logspace(np.log10(x.min()), np.log10(d_fit_min), 100)
        y_extrap = 10 ** (fit.intercept + fit.slope * np.log10(x_extrap))
        plt.plot(x_extrap, y_extrap, linestyle=":", color=color, alpha=0.35)
