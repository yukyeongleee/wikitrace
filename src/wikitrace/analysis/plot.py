"""Figure of Gain and Regression per method with 95% intervals and the no-memory floor."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .gain_regression import FLOOR

METHOD_COLOUR = {
    "awm": "#F5C796", "expel": "#D9700F", "reasoningbank": "#E6D29A",
    "00I-inject": "#8FD7D7", "00I-traverse": "#00A3B0", "RCI-traverse": "#0A5F68",
}
OURS = "RCI-traverse"
DISPLAY = {
    "no-memory": "No Memory", "00I-inject": "Knowledge Injection", OURS: "WikiTrace", "awm": "AWM",
    "expel": "ExpeL", "reasoningbank": "ReasoningBank",
}
NEUTRAL = "#B8BEC4"
INK, MUTED, GRID = "#20262E", "#65717A", "#D9E0E5"
COLUMN_WIDTH, TEXT_WIDTH = 3.3, 6.9
LABEL = {"gain": "Gain (%)", "regression": "Regression (%)"}
RC = {
    "font.family": "sans-serif", "font.sans-serif": ["TeX Gyre Heros", "Nimbus Sans", "Helvetica", "Arial",
                                                     "DejaVu Sans"],
    "font.size": 9.5, "axes.labelsize": 9.5, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "axes.edgecolor": INK, "axes.linewidth": 0.7, "text.color": INK, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.major.pad": 2, "ytick.major.pad": 2,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5, "axes.labelpad": 2,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.45, "axes.axisbelow": True,
    "pdf.fonttype": 42, "savefig.dpi": 300,
}


def display(method: str) -> str:
    """The paper's name for a method label; an unknown label unchanged."""
    return "WikiTrace" if method.startswith("RCI-preload-usermem-descend-sourcecheck") else DISPLAY.get(method, method)


def panel(ax, t: pd.DataFrame, methods: list[str], xlim: tuple[float, float]) -> None:
    """One metric on one unit: a row per method and the floor as a dashed line over its interval."""
    by_method = t.set_index("method")
    if FLOOR in by_method.index:
        f = by_method.loc[FLOOR]
        ax.axvspan(f.ci_lo, f.ci_hi, color=GRID, alpha=0.6, lw=0, zorder=1)
        ax.axvline(f.estimate, color=INK, linestyle=(0, (4, 2)), linewidth=0.9, zorder=2)
    for y, method in zip(range(len(methods) - 1, -1, -1), methods):
        if method not in by_method.index:
            continue
        r, ours = by_method.loc[method], display(method) == "WikiTrace"
        ax.plot([r.ci_lo, r.ci_hi], [y, y], color=INK if ours else MUTED, lw=0.8, zorder=3)
        ax.scatter(r.estimate, y, s=42, facecolor=METHOD_COLOUR.get(OURS if ours else method, NEUTRAL), zorder=4,
                   edgecolor=INK if ours else MUTED, linewidth=1.1 if ours else 0.5)
    pad = 0.03 * (xlim[1] - xlim[0])
    ax.set_xlim(xlim[0] - pad, xlim[1] + pad)
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0, pad=4)
    ax.set_ylim(-0.6, len(methods) + 0.5)
    ax.set_yticks(range(len(methods) - 1, -1, -1), [display(m) for m in methods])


def limit(t: pd.DataFrame) -> float:
    return next(x for x in (5, 10, 20, 40, 50, 80, 100) if x >= t.ci_hi.max())


def save(fig, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".pdf", ".png"):
        fig.savefig(out.with_suffix(suffix), bbox_inches="tight", pad_inches=0.02)


def plot(stats_dir: str | Path, out: str | Path, methods: list[str] | None = None) -> list[Path]:
    """Write the suite-average figure to out and the per-suite figure beside it, each as PDF and PNG."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(RC)
    stats_dir, out = Path(stats_dir), Path(out)
    avg = pd.read_csv(stats_dir / "gain_regression_average.csv")
    suite = pd.read_csv(stats_dir / "gain_regression_suite.csv")
    methods = methods or [a for a in dict.fromkeys(avg.method) if a != FLOOR]

    fig, axes = plt.subplots(1, 2, figsize=(COLUMN_WIDTH, 0.35 * len(methods) + 0.4), sharey=True,
                             gridspec_kw={"wspace": 0.12})
    for ax, metric in zip(axes, LABEL):
        t = avg[avg.metric == metric]
        panel(ax, t, methods, (0, limit(t)))
        ax.set_xlabel(LABEL[metric])
    save(fig, out)

    names = list(dict.fromkeys(suite.suite))
    cols = min(4, len(names))
    rows = -(-len(names) // cols)
    per_suite = out.with_name(out.stem + "_suite")
    fig, axes = plt.subplots(2 * rows, cols, figsize=(TEXT_WIDTH, 0.35 * len(methods) * 2 * rows + 1),
                             sharey=True, squeeze=False, gridspec_kw={"wspace": 0.12, "hspace": 0.6})
    for block, metric in enumerate(LABEL):
        t = suite[suite.metric == metric]
        lim = limit(t)
        for i, name in enumerate(names):
            ax = axes[block * rows + i // cols][i % cols]
            panel(ax, t[t.suite == name], methods, (0, lim))
            ax.set_xticks([0, lim / 2])
            ax.set_title(name, fontsize=8.5, pad=3)
            ax.set_xlabel(LABEL[metric], fontsize=8)
        for j in range(len(names), rows * cols):
            axes[block * rows + j // cols][j % cols].remove()
    save(fig, per_suite)
    return [out.with_suffix(".pdf"), per_suite.with_suffix(".pdf")]
