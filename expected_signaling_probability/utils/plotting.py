from expected_signaling_probability.utils.fitting import fit_power_law
from expected_signaling_probability.utils.stats import Stats
from matplotlib.ticker import ScalarFormatter, NullFormatter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from enum import Enum
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import shutil
import os


class PlotMode(Enum):
    EXPLORE = "explore"
    PAPER = "paper"


@dataclass(frozen=True)
class _StyleConfig:
    fig_width: float
    fig_aspect: float
    font_size: float
    legend_fontsize: float
    marker_size: float
    capsize: float
    font_scale: float
    axes_linewidth: float
    grid_linewidth: float
    lines_linewidth: float
    save_format: str
    show_title: bool


_STYLES: dict[PlotMode, _StyleConfig] = {
    PlotMode.EXPLORE: _StyleConfig(
        fig_width=10,
        fig_aspect=0.6,
        font_size=12,
        legend_fontsize=10,
        marker_size=50,
        capsize=3,
        font_scale=1.0,
        axes_linewidth=1.2,
        grid_linewidth=0.7,
        lines_linewidth=1.5,
        save_format="png",
        show_title=True,
    ),
    PlotMode.PAPER: _StyleConfig(
        fig_width=3.4,
        fig_aspect=0.75,
        font_size=10,
        legend_fontsize=8,
        marker_size=12,
        capsize=2,
        font_scale=1.0,
        axes_linewidth=0.8,
        grid_linewidth=0.5,
        lines_linewidth=1.0,
        save_format="pdf",
        show_title=False,
    ),
}

_active_config: _StyleConfig = _STYLES[PlotMode.EXPLORE]

PLOTS_DIR = Path("plots")


def apply_plot_style(mode: PlotMode = PlotMode.EXPLORE):
    global _active_config
    _active_config = _STYLES[mode]
    cfg = _active_config

    sns.set_theme(
        style="whitegrid",
        context="notebook",
        font_scale=cfg.font_scale,
    )

    use_tex = shutil.which("pdflatex") is not None
    if use_tex:
        os.environ["PATH"] = "/Library/TeX/texbin:" + os.environ["PATH"]

    plt.rcParams.update(
        {
            "figure.figsize": (cfg.fig_width, cfg.fig_width * cfg.fig_aspect),
            "figure.dpi": 150,
            "font.size": cfg.font_size,
            "axes.titlesize": cfg.font_size,
            "axes.labelsize": cfg.font_size,
            "xtick.labelsize": cfg.font_size - 1,
            "ytick.labelsize": cfg.font_size - 1,
            "legend.fontsize": cfg.legend_fontsize,
            "axes.linewidth": cfg.axes_linewidth,
            "grid.linewidth": cfg.grid_linewidth,
            "grid.alpha": 0.3,
            "lines.linewidth": cfg.lines_linewidth,
            "axes.unicode_minus": False,
            "axes.labelpad": 2 if mode == PlotMode.PAPER else 4,
            "xtick.major.pad": 3.5 if mode == PlotMode.PAPER else 3.5,
            "ytick.major.pad": 2 if mode == PlotMode.PAPER else 3.5,
            "xtick.major.size": 3 if mode == PlotMode.PAPER else 3.5,
            "ytick.major.size": 3 if mode == PlotMode.PAPER else 3.5,
            "xtick.direction": "in" if mode == PlotMode.PAPER else "out",
            "ytick.direction": "in" if mode == PlotMode.PAPER else "out",
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02 if mode == PlotMode.PAPER else 0.1,
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


def save_plot(name: str, fmt: str | None = None):
    fmt = fmt or _active_config.save_format
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = PLOTS_DIR / f"{name}_{timestamp}.{fmt}"
    plt.savefig(filename)
    print(f"[save_plot] Saved to {filename}")


def plot_title(title: str):
    if _active_config.show_title:
        plt.title(title)


def format_log_ticks():
    ax = plt.gca()
    formatter = ScalarFormatter()
    formatter.set_scientific(False)
    ax.xaxis.set_major_formatter(formatter)
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))  # type: ignore[attr-defined]


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
    fill_kwargs = (
        {"color": color}
        if _active_config.show_title
        else {"facecolors": "none", "edgecolors": color, "linewidths": 0.8}
    )
    plt.scatter(
        x, y, s=_active_config.marker_size, zorder=2,
        label=rf"Data: {label}" if _active_config.show_title else "_nolegend_",
        **fill_kwargs,  # type: ignore[arg-type]
    )


def plot_error_bars(
    x: np.typing.NDArray,
    y: np.typing.NDArray,
    all_stats: list[Stats],
):
    yerr_low = y - np.array([s.q25 for s in all_stats])
    yerr_high = np.array([s.q75 for s in all_stats]) - y
    elinewidth = 0.6 if not _active_config.show_title else 1.0
    plt.errorbar(
        x,
        y,
        yerr=[yerr_low, yerr_high],
        label=r"IQR" if _active_config.show_title else "_nolegend_",
        fmt="none",
        capsize=_active_config.capsize,
        ecolor="grey",
        elinewidth=elinewidth,
        capthick=elinewidth,
        zorder=1,
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

    if _active_config.show_title:
        fit_range = rf" (${dim_var} \geq {d_fit_min}$)" if d_fit_min is not None else ""
        fit_label = rf"Fit{fit_range}: {probability_label} $ \propto {dim_var}^{{{fit.slope:.2f} \pm {fit.stderr:.2f}}}$"
    else:
        fit_label = rf"{probability_label} $ \propto {dim_var}^{{{fit.slope:.2f} \pm {fit.stderr:.2f}}}$"

    plt.plot(
        fit.x_fit,
        fit.y_fit,
        label=fit_label,
        linestyle="--",
        color=color,
        zorder=3,
    )

    if d_fit_min is not None and x.min() < d_fit_min:
        x_extrap = np.logspace(np.log10(x.min()), np.log10(d_fit_min), 100)
        y_extrap = 10 ** (fit.intercept + fit.slope * np.log10(x_extrap))
        plt.plot(x_extrap, y_extrap, linestyle=":", color=color, alpha=0.35)
