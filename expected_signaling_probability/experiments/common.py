"""Shared declarations and plotting for the paper figures.

Every figure script declares its runs as ``DrawConfig`` sweeps, pulls the
samples from the cache under ``data/runs`` (computing whatever is missing),
and saves ``plots/<name>.pdf`` and ``plots/<name>.png``.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import numpy as np

from expected_signaling_probability import Direction, DrawConfig, GlobalOpEnsemble, Samples, summary_table
from expected_signaling_probability.utils.plotting import (
    PlotMode,
    apply_plot_style,
    format_log_ticks,
    plot_error_bars,
    plot_power_law_fit,
    plot_scatter,
    plot_title,
    save_plot,
)

MASTER_SEED = 0
N_SAMPLES = 1_000

GLOBAL_OP_LABEL = {
    GlobalOpEnsemble.HAAR_UNITARY: "unitary",
    GlobalOpEnsemble.COMS: "COMS",
    GlobalOpEnsemble.BCSZ: "BCSZ",
    GlobalOpEnsemble.WEAK_COUPLING: "weak coupling",
}
GLOBAL_OP_COLOR = {
    GlobalOpEnsemble.HAAR_UNITARY: "tab:blue",
    GlobalOpEnsemble.COMS: "tab:green",
    GlobalOpEnsemble.BCSZ: "tab:red",
    GlobalOpEnsemble.WEAK_COUPLING: "tab:purple",
}
PAPER_GLOBAL_OPS = (GlobalOpEnsemble.HAAR_UNITARY, GlobalOpEnsemble.COMS, GlobalOpEnsemble.BCSZ)


def kraus_rank(global_op: GlobalOpEnsemble, d_A: int, d_B: int, rank: int | None = None) -> int:
    """Kraus rank K of the global ensemble (environment dimension of its Stinespring dilation)."""
    D = d_A * d_B
    match global_op:
        case GlobalOpEnsemble.HAAR_UNITARY:
            return 1
        case GlobalOpEnsemble.COMS:
            return D
        case GlobalOpEnsemble.BCSZ:
            return D * D if rank is None else rank
    raise ValueError(global_op)


def page_exponent(global_op: GlobalOpEnsemble, direction: Direction, d_B: int, rank: int | None = None) -> float | None:
    """Predicted large-d_A power of S and T at fixed d_B from the Page rule S, T ~ sqrt(d_receiver / d_discarded).

    d_discarded = d_sender * K with K the Kraus rank. Returns None when the rule predicts saturation.
    """
    # K as a power of d_A at fixed d_B: unitary 0, COMS 1, full-rank BCSZ 2, fixed rank 0.
    k_power = {GlobalOpEnsemble.HAAR_UNITARY: 0, GlobalOpEnsemble.COMS: 1, GlobalOpEnsemble.BCSZ: 2 if rank is None else 0}[global_op]
    receiver_power = 0 if direction is Direction.A_TO_B else 1
    sender_power = 1 if direction is Direction.A_TO_B else 0
    power = 0.5 * (receiver_power - sender_power - k_power)
    return None if power >= 0 else power


class ErrorBar:
    SEM = "sem"
    IQR = "iqr"
    NONE = "none"


@dataclass
class Curve:
    """One line on a plot: a list of ``Samples`` ordered along the x axis."""

    label: str
    color: str
    samples: list[Samples]
    x_attr: str = "d_A"
    marker: str = "o"
    linestyle: str = "--"
    fit: bool = True
    y_stat: str = "mean"  # "mean" | "median"

    @property
    def x(self) -> np.ndarray:
        return np.array([getattr(s.config, self.x_attr) for s in self.samples], dtype=float)

    @property
    def y(self) -> np.ndarray:
        return np.array([getattr(s, self.y_stat) for s in self.samples])


@dataclass
class Figure:
    name: str
    curves: list[Curve]
    xlabel: str
    ylabel: str
    dim_var: str = "d"
    title: str = ""
    d_fit_min: int | None = None
    error_bar: str = ErrorBar.SEM
    mode: PlotMode = PlotMode.PAPER
    save: bool = True
    show: bool = False
    log_y: bool = True
    extra: dict = field(default_factory=dict)


def draw_curves(curves: list[Curve], *, error_bar: str, dim_var: str, d_fit_min: int | None, log_y: bool, xlabel: str, ylabel: str) -> None:
    """Draw ``curves`` on the current axes."""
    for k, c in enumerate(curves):
        x, y = c.x, c.y
        err_label = "" if k else error_bar.upper()
        plot_scatter(x, y, color=c.color, label=c.label, marker=c.marker)
        if c.fit:
            plot_power_law_fit(x, y, color=c.color, probability_label=c.label, dim_var=dim_var, d_fit_min=d_fit_min)
        else:
            plt.plot(x, y, linestyle=c.linestyle, color=c.color, label=c.label, alpha=0.8)
        if error_bar == ErrorBar.SEM:
            err = np.array([s.sem for s in c.samples])
            plot_error_bars(x, y, err, err, label=err_label, color=c.color)
        elif error_bar == ErrorBar.IQR:
            lo = y - np.array([s.q25 for s in c.samples])
            hi = np.array([s.q75 for s in c.samples]) - y
            plot_error_bars(x, y, lo, hi, label=err_label, color=c.color)
    plt.xscale("log")
    if log_y:
        plt.yscale("log")
    format_log_ticks()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)


def plot_figure(fig: Figure) -> None:
    apply_plot_style(fig.mode)
    plt.figure()
    draw_curves(fig.curves, error_bar=fig.error_bar, dim_var=fig.dim_var, d_fit_min=fig.d_fit_min, log_y=fig.log_y, xlabel=fig.xlabel, ylabel=fig.ylabel)
    if fig.title:
        plot_title(fig.title)
    plt.legend(loc="lower left")
    plt.tight_layout()
    if fig.save:
        save_plot(fig.name, timestamped=fig.mode is PlotMode.EXPLORE)
    if fig.show:
        plt.show()
    else:
        plt.close()


def print_summary(curves: list[Curve]) -> None:
    import pandas as pd

    with pd.option_context("display.width", 250, "display.max_columns", 30):
        for c in curves:
            print(f"\n== {c.label}")
            df = summary_table(c.samples)
            print(df[["d_A", "d_B", "direction", "n", "mean", "sem", "median", "q25", "q75"]].to_string(index=False))


def symmetric_configs(base: DrawConfig, d_min: int, d_max: int) -> list[DrawConfig]:
    return [base.with_(d_A=d, d_B=d) for d in range(d_min, d_max + 1)]


def asymmetric_configs(base: DrawConfig, d_A_min: int, d_A_max: int, d_B: int) -> list[DrawConfig]:
    return [base.with_(d_A=d_A, d_B=d_B) for d_A in range(d_A_min, d_A_max + 1)]
