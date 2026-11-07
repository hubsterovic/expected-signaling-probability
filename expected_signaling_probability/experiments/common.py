"""Shared declarations and plotting for the paper experiments."""
from __future__ import annotations

from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import numpy as np

from expected_signaling_probability import DrawConfig, GlobalOpEnsemble, Samples, summary_table
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

N_SAMPLES = 1_000
MASTER_SEED = 0

GLOBAL_OP_LABEL = {
    GlobalOpEnsemble.HAAR_UNITARY: "unitary",
    GlobalOpEnsemble.COMS: r"COMS",
    GlobalOpEnsemble.BCSZ: "BCSZ",
    GlobalOpEnsemble.WEAK_COUPLING: "weak coupling",
}
GLOBAL_OP_COLOR = {
    GlobalOpEnsemble.HAAR_UNITARY: "tab:blue",
    GlobalOpEnsemble.COMS: "tab:green",
    GlobalOpEnsemble.BCSZ: "tab:red",
    GlobalOpEnsemble.WEAK_COUPLING: "tab:purple",
}


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
    show: bool = True
    log_y: bool = True
    extra: dict = field(default_factory=dict)


def plot_figure(fig: Figure) -> None:
    apply_plot_style(fig.mode)
    plt.figure()
    for k, c in enumerate(fig.curves):
        x, y = c.x, c.y
        err_label = "" if k else fig.error_bar.upper()
        plot_scatter(x, y, color=c.color, label=c.label)
        if c.fit:
            plot_power_law_fit(x, y, color=c.color, probability_label=c.label, dim_var=fig.dim_var, d_fit_min=fig.d_fit_min)
        else:
            plt.plot(x, y, linestyle=c.linestyle, color=c.color, label=c.label, alpha=0.8)
        if fig.error_bar == ErrorBar.SEM:
            err = np.array([s.sem for s in c.samples])
            plot_error_bars(x, y, err, err, label=err_label, color=c.color)
        elif fig.error_bar == ErrorBar.IQR:
            lo = y - np.array([s.q25 for s in c.samples])
            hi = np.array([s.q75 for s in c.samples]) - y
            plot_error_bars(x, y, lo, hi, label=err_label, color=c.color)
    plt.xscale("log")
    if fig.log_y:
        plt.yscale("log")
    format_log_ticks()
    plt.xlabel(fig.xlabel)
    plt.ylabel(fig.ylabel)
    if fig.title:
        plot_title(fig.title)
    plt.legend()
    plt.tight_layout()
    if fig.save:
        save_plot(fig.name)
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
