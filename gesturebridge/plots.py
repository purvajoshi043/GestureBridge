"""Report charts (matplotlib PNGs) in one consistent, accessible style.

Each model keeps the same colour in every chart (colour follows the entity):
Random Forest = blue, SVM = orange, MLP = aqua (a palette validated for
colour-blind readers). Bars start at zero with rounded data-ends, gridlines are
hairlines, text uses ink colours, and every chart has a table twin in the
dashboard / report.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.patches import PathPatch  # noqa: E402
from matplotlib.path import Path as MplPath  # noqa: E402

DPI = 150
BAR_PX, GAP_PX = 34, 3   # bar thickness cap and the surface gap between touching bars (device pixels)
SURFACE = "#fcfcfb"
INK, INK_2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
MODEL_COLOURS = {"rf": SERIES[0], "svm": SERIES[1], "mlp": SERIES[2]}
MODEL_NAMES = {"rf": "Random Forest", "svm": "SVM (RBF kernel)", "mlp": "MLP neural network"}
BLUE_RAMP = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
             "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
SEQUENTIAL = LinearSegmentedColormap.from_list("gesturebridge_blue", [SURFACE] + BLUE_RAMP)


def apply_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "figure.dpi": DPI, "savefig.dpi": DPI,
        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "Helvetica Neue", "Arial", "DejaVu Sans"],
        "font.size": 9.5,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
        "axes.labelcolor": INK_2, "axes.titlecolor": INK, "axes.titlesize": 11.5,
        "axes.titleweight": "semibold", "axes.titlelocation": "left", "axes.titlepad": 12,
        "axes.spines.top": False, "axes.spines.right": False,
        "xtick.color": AXIS, "ytick.color": AXIS, "xtick.labelcolor": INK_2, "ytick.labelcolor": INK_2,
        "grid.color": GRID, "grid.linewidth": 0.7, "grid.linestyle": "-",
        "legend.frameon": False, "legend.labelcolor": INK_2, "legend.fontsize": 9,
    })


def _px_to_data(ax) -> tuple[float, float]:
    ax.figure.canvas.draw()
    box = ax.get_window_extent()
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    return abs(x1 - x0) / box.width, abs(y1 - y0) / box.height


def round_bar_ends(ax, bars, horizontal: bool = False, radius_px: float = 4.0) -> None:
    """Redraw bars with 4 px rounded data-ends and a square baseline. Call after limits are final."""
    dx, dy = _px_to_data(ax)
    C3, L = MplPath.CURVE3, MplPath.LINETO
    for bar in bars:
        x, y, w, h = bar.get_x(), bar.get_y(), bar.get_width(), bar.get_height()
        if w <= 0 or h <= 0:
            continue
        if horizontal:
            rx, ry = min(radius_px * dx, w), min(radius_px * dy, h / 2)
            verts = [(x, y), (x + w - rx, y), (x + w, y), (x + w, y + ry), (x + w, y + h - ry),
                     (x + w, y + h), (x + w - rx, y + h), (x, y + h), (x, y)]
        else:
            rx, ry = min(radius_px * dx, w / 2), min(radius_px * dy, h)
            verts = [(x, y), (x, y + h - ry), (x, y + h), (x + rx, y + h), (x + w - rx, y + h),
                     (x + w, y + h), (x + w, y + h - ry), (x + w, y), (x, y)]
        codes = [MplPath.MOVETO, L, C3, C3, L, C3, C3, L, MplPath.CLOSEPOLY]
        ax.add_patch(PathPatch(MplPath(verts, codes), facecolor=bar.get_facecolor(),
                               edgecolor="none", zorder=bar.get_zorder()))
        bar.set_visible(False)


def _finish(fig, path: Path) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    return str(path)


def model_comparison(results: dict[str, dict[str, float]], path: Path) -> str:
    """Grouped bars: test accuracy, test macro-F1 and cross-validated macro-F1 per model."""
    apply_style()
    metrics = [("accuracy", "Test accuracy"), ("macro_f1", "Test macro-F1"), ("cv_macro_f1", "Cross-val macro-F1")]
    models = list(results)
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    ax.set_xlim(-0.5, len(metrics) - 0.5)
    ax.set_ylim(0, 1.08)
    dx, _ = _px_to_data(ax)
    width, gap = min(0.24, BAR_PX * dx), GAP_PX * dx
    all_bars, tops = [], []
    for i, model in enumerate(models):
        offsets = np.arange(len(metrics)) + (i - (len(models) - 1) / 2) * (width + gap)
        values = [results[model].get(key, np.nan) for key, _ in metrics]
        bars = ax.bar(offsets, np.nan_to_num(values), width, color=MODEL_COLOURS.get(model, SERIES[i]),
                      label=MODEL_NAMES.get(model, model), zorder=3)
        all_bars += list(bars)
        tops.append(values)
    ax.set_xticks(range(len(metrics)), [label for _, label in metrics])
    ax.set_yticks(np.linspace(0, 1, 6), [f"{v:.0%}" for v in np.linspace(0, 1, 6)])
    ax.yaxis.grid(True, zorder=0)
    ax.tick_params(axis="x", length=0)
    ax.set_title("Model comparison on held-out recordings")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=len(models))
    tops = np.array(tops, dtype=float)
    for j in range(len(metrics)):  # label only the best model per metric
        if np.all(np.isnan(tops[:, j])):
            continue
        i = int(np.nanargmax(tops[:, j]))
        x = j + (i - (len(models) - 1) / 2) * (width + gap)
        ax.annotate(f"{tops[i, j]:.1%}", (x, tops[i, j]), xytext=(0, 4), textcoords="offset points",
                    ha="center", va="bottom", color=INK, fontsize=8.5, fontweight="semibold")
    round_bar_ends(ax, all_bars)
    return _finish(fig, path)


def latency_chart(latency_ms: dict[str, float], path: Path) -> str:
    apply_style()
    models = list(latency_ms)
    fig, ax = plt.subplots(figsize=(5.6, 2.6))
    values = [latency_ms[m] for m in models]
    ax.set_ylim(len(models) - 0.5, -0.5)
    _, dy = _px_to_data(ax)
    bars = ax.barh(range(len(models)), values, min(0.5, BAR_PX * dy),
                   color=[MODEL_COLOURS.get(m, SERIES[i]) for i, m in enumerate(models)], zorder=3)
    ax.set_yticks(range(len(models)), [MODEL_NAMES.get(m, m) for m in models])
    ax.set_xlim(0, max(values) * 1.25 if max(values) > 0 else 1)
    ax.xaxis.grid(True, zorder=0)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("Milliseconds per frame (one prediction)")
    ax.set_title("Prediction time per frame")
    for i, v in enumerate(values):
        ax.annotate(f"{v:.2f} ms", (v, i), xytext=(4, 0), textcoords="offset points", va="center",
                    color=INK_2, fontsize=8.5)
    round_bar_ends(ax, bars, horizontal=True)
    return _finish(fig, path)


def confusion_heatmap(normalised: np.ndarray, labels: list[str], title: str, path: Path) -> str:
    """Row-normalised confusion matrix; the diagonal and confusions >= 5% are labelled."""
    apply_style()
    n = len(labels)
    size = max(5.5, 0.38 * n + 2.2)
    fig, ax = plt.subplots(figsize=(size + 1.0, size))
    image = ax.imshow(normalised, cmap=SEQUENTIAL, vmin=0, vmax=1, interpolation="nearest")
    names = [label.replace("_", " ") for label in labels]
    ax.set_xticks(range(n), names, rotation=90)
    ax.set_yticks(range(n), names)
    ax.set_xlabel("Predicted sign")
    ax.set_ylabel("True sign")
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for i in range(n):
        for j in range(n):
            value = normalised[i, j]
            if (i == j and value > 0) or value >= 0.05:
                ax.text(j, i, f"{value * 100:.0f}", ha="center", va="center", fontsize=7,
                        color="#ffffff" if value > 0.55 else INK)
    bar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.02)
    bar.outline.set_visible(False)
    bar.ax.tick_params(color=AXIS, labelcolor=INK_2, length=0)
    bar.set_label("Share of the true sign's samples", color=INK_2)
    bar.set_ticks([0, 0.25, 0.5, 0.75, 1], labels=["0%", "25%", "50%", "75%", "100%"])
    ax.set_title(title)
    return _finish(fig, path)


def horizontal_bars(values: dict[str, float], title: str, xlabel: str, path: Path,
                    colour: str = SERIES[0], percent: bool = True, sort: bool = True) -> str:
    """One series of horizontal bars (per-class F1, feature importance ...)."""
    apply_style()
    items = sorted(values.items(), key=lambda kv: kv[1]) if sort else list(values.items())
    names = [k.replace("_", " ") for k, _ in items]
    data = [v for _, v in items]
    fig, ax = plt.subplots(figsize=(6.4, max(2.6, 0.27 * len(items) + 1.0)))
    ax.set_ylim(-0.5, len(items) - 0.5)
    _, dy = _px_to_data(ax)
    bars = ax.barh(range(len(items)), data, min(0.62, BAR_PX * dy), color=colour, zorder=3)
    ax.set_yticks(range(len(items)), names)
    ax.tick_params(axis="y", length=0)
    top = max(data) if data else 1
    ax.set_xlim(0, 1.12 if percent else top * 1.18)
    if percent:
        ax.set_xticks(np.linspace(0, 1, 6), [f"{v:.0%}" for v in np.linspace(0, 1, 6)])
    ax.xaxis.grid(True, zorder=0)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    for i, v in enumerate(data):
        ax.annotate(f"{v:.0%}" if percent else f"{v:.3f}", (v, i), xytext=(3, 0), textcoords="offset points",
                    va="center", color=INK_2, fontsize=7.5)
    round_bar_ends(ax, bars, horizontal=True)
    return _finish(fig, path)


def grouped_by_signer(loso: dict[str, dict[str, float]], path: Path) -> str:
    """Leave-one-signer-out accuracy: one group per held-out signer, one bar per model."""
    apply_style()
    models = list(loso)
    signers = sorted({s for scores in loso.values() for s in scores})
    fig, ax = plt.subplots(figsize=(max(5.5, 1.5 * len(signers) + 2.5), 3.6))
    ax.set_xlim(-0.5, len(signers) - 0.5)
    ax.set_ylim(0, 1.05)
    dx, _ = _px_to_data(ax)
    width, gap = min(0.24, BAR_PX * dx), GAP_PX * dx
    all_bars = []
    for i, model in enumerate(models):
        offsets = np.arange(len(signers)) + (i - (len(models) - 1) / 2) * (width + gap)
        values = [loso[model].get(s, 0.0) for s in signers]
        all_bars += list(ax.bar(offsets, values, width, color=MODEL_COLOURS.get(model, SERIES[i]),
                                label=MODEL_NAMES.get(model, model), zorder=3))
    ax.set_xticks(range(len(signers)), [f"Tested on {s}" for s in signers])
    ax.set_yticks(np.linspace(0, 1, 6), [f"{v:.0%}" for v in np.linspace(0, 1, 6)])
    ax.yaxis.grid(True, zorder=0)
    ax.tick_params(axis="x", length=0)
    ax.set_ylabel("Accuracy on the unseen signer")
    ax.set_title("Generalisation to a new person (leave-one-signer-out)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=len(models))
    round_bar_ends(ax, all_bars)
    return _finish(fig, path)


def decoder_robustness(levels: list[float], series: dict[str, list[float]], path: Path) -> str:
    """Sentence accuracy vs simulated sign accuracy for greedy and beam-search decoding."""
    apply_style()
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    xs = [level * 100 for level in levels]
    for i, (name, values) in enumerate(series.items()):
        ys = [v * 100 for v in values]
        ax.plot(xs, ys, color=SERIES[i], linewidth=2, solid_capstyle="round", solid_joinstyle="round",
                marker="o", markersize=8, markerfacecolor=SERIES[i], markeredgecolor=SURFACE,
                markeredgewidth=2, label=name, zorder=3)
        ax.annotate(f"{name} {ys[-1]:.0f}%", (xs[-1], ys[-1]), xytext=(8, 0), textcoords="offset points",
                    va="center", color=INK_2, fontsize=8.5)
    ax.set_xlim(max(xs) + 2, min(xs) - 14)   # left = cleanest input, right = noisiest
    ax.set_ylim(0, 105)
    ax.yaxis.grid(True, zorder=0)
    ax.set_xlabel("Simulated sign accuracy (top-1), %")
    ax.set_ylabel("Correct sentences, %")
    ax.set_title("Beam search keeps sentences correct when sign recognition is noisy")
    ax.legend(loc="lower left")
    return _finish(fig, path)
