"""One consistent, print-friendly chart style for every figure in the report."""
from __future__ import annotations

from pathlib import Path

# Categorical colours, always assigned in this fixed order (validated for colour-vision deficiency).
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SEQUENTIAL = ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"


def seq_cmap():
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list("rl_blues", SEQUENTIAL)


def apply_style() -> None:
    import matplotlib as mpl

    mpl.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "figure.dpi": 110,
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.labelcolor": INK_2,
        "axes.edgecolor": GRID,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "text.color": INK,
        "lines.linewidth": 2,
        "lines.markersize": 5,
        "legend.frameon": False,
        "axes.prop_cycle": mpl.cycler(color=PALETTE),
    })


def save_fig(fig, path_no_ext: str | Path) -> None:
    """Save as 300-dpi PNG (slides/notebook) and PDF (vector, for the LaTeX/Word report)."""
    import matplotlib.pyplot as plt

    p = Path(path_no_ext)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(p.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(p.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
