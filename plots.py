"""Shared plots. Saves PNG and PDF at 300 dpi."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .metrics import reliability_table, risk_coverage

COLORS = {"Decision 1": "#2a6fdb", "Decision 1 (TS)": "#1b3f7a",
          "DeBERTa-v3": "#d9822b", "DeBERTa-v3 (TS)": "#8c5a2b"}


def save(fig, path_no_ext):
    for ext in ("png", "pdf"):
        fig.savefig(f"{path_no_ext}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def reliability_panel(ax, systems, title, n_bins=15, min_count=5):
    """systems: dict name -> (probs, labels). Bins with fewer than min_count items are hidden."""
    ax.plot([0, 1], [0, 1], ls="--", c="grey", lw=1)
    for name, (p, y) in systems.items():
        rows = [r for r in reliability_table(p, y, n_bins) if r["n"] >= min_count]
        ax.plot([r["mean_conf"] for r in rows], [r["acc"] for r in rows], marker="o", ms=4,
                label=name, c=COLORS.get(name))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_title(title)
    ax.legend(fontsize=8, loc="upper left")


def coverage_panel(ax, systems, title):
    for name, (p, y) in systems.items():
        cov, acc = risk_coverage(p, y)
        ax.plot(cov, acc, label=name, c=COLORS.get(name))
    ax.set_xlabel("Coverage")
    ax.set_ylabel("Accuracy")
    ax.set_title(title)
    ax.legend(fontsize=8)


def bar_with_ci(ax, labels, vals, los, his, ylabel):
    vals, los, his = map(np.asarray, (vals, los, his))
    ax.bar(labels, vals, yerr=[vals - los, his - vals], capsize=4, color="#2a6fdb")
    ax.set_ylabel(ylabel)
