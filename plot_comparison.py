#!/usr/bin/env python3
"""Plot base vs trained model comparison results."""

import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from collections import defaultdict

ANALYSIS_FILE = r"C:\coding vs\syntax_error_rl\data\comparison_analysis.json"
OUTPUT_FILE = r"C:\coding vs\syntax_error_rl\data\comparison_plot.png"

# ── Palette (validated default, light mode) ──────────────────────────────
SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
TEXT_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"
SERIES_1 = "#2a78d6"   # blue — base
SERIES_2 = "#eb6834"   # orange — trained
GOOD = "#0ca30c"
CRITICAL = "#d03b3b"

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "axes.edgecolor": BASELINE,
    "axes.labelcolor": TEXT_SECONDARY,
    "text.color": TEXT_PRIMARY,
    "xtick.color": TEXT_MUTED,
    "ytick.color": TEXT_MUTED,
    "grid.color": GRIDLINE,
    "grid.linewidth": 0.8,
    "axes.linewidth": 0.8,
    "font.family": "system-ui, -apple-system, 'Segoe UI', sans-serif",
    "font.size": 11,
})


def load_analysis():
    with open(ANALYSIS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def plot_per_type(ax, per_type):
    """Grouped bar chart: base vs trained accuracy per question type."""
    types = sorted(per_type.keys())
    base_acc = [per_type[t]["base_acc"] * 100 for t in types]
    trained_acc = [per_type[t]["trained_acc"] * 100 for t in types]

    x = np.arange(len(types))
    width = 0.38

    bars1 = ax.bar(x - width / 2, base_acc, width, color=SERIES_1,
                   label="Base", zorder=3, linewidth=0)
    bars2 = ax.bar(x + width / 2, trained_acc, width, color=SERIES_2,
                   label="Trained", zorder=3, linewidth=0)

    # Direct labels on bars
    for bar in bars1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 1.5, f"{h:.0f}",
                ha="center", va="bottom", fontsize=8.5, color=TEXT_SECONDARY)
    for bar in bars2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 1.5, f"{h:.0f}",
                ha="center", va="bottom", fontsize=8.5, color=TEXT_SECONDARY)

    ax.set_xticks(x)
    ax.set_xticklabels(types, rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 110)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(decimals=0))
    ax.grid(axis="y", zorder=0)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper left", fontsize=10)
    ax.set_title("Per-type accuracy: base vs trained", fontsize=12,
                 fontweight="bold", pad=12, color=TEXT_PRIMARY)

    # Remove top/right spines
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def plot_learned_forgotten(ax, learned, forgotten):
    """Bar chart: learned vs forgotten counts."""
    categories = ["Learned\n(wrong → right)", "Forgotten\n(right → wrong)"]
    values = [learned, forgotten]
    colors = [GOOD, CRITICAL]

    bars = ax.bar(categories, values, color=colors, width=0.5, zorder=3, linewidth=0)

    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 1, f"{int(h)}",
                ha="center", va="bottom", fontsize=12, fontweight="bold",
                color=TEXT_PRIMARY)

    ax.set_ylabel("Number of questions")
    ax.set_ylim(0, max(values) * 1.25)
    ax.grid(axis="y", zorder=0)
    ax.set_axisbelow(True)
    ax.set_title("What the model learned vs forgot", fontsize=12,
                 fontweight="bold", pad=12, color=TEXT_PRIMARY)

    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def plot_overall(ax, base_acc, trained_acc):
    """Overall accuracy comparison."""
    categories = ["Base", "Trained"]
    values = [base_acc * 100, trained_acc * 100]
    colors = [SERIES_1, SERIES_2]

    bars = ax.bar(categories, values, color=colors, width=0.5, zorder=3, linewidth=0)

    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 1.5, f"{h:.1f}%",
                ha="center", va="bottom", fontsize=12, fontweight="bold",
                color=TEXT_PRIMARY)

    # Delta annotation
    delta = (trained_acc - base_acc) * 100
    ax.annotate(f"+{delta:.1f} pts", xy=(1, trained_acc * 100),
                xytext=(0.5, trained_acc * 100 + 12),
                ha="center", fontsize=11, fontweight="bold", color=GOOD,
                arrowprops=dict(arrowstyle="->", color=GOOD, lw=1.5))

    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 100)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(decimals=0))
    ax.grid(axis="y", zorder=0)
    ax.set_axisbelow(True)
    ax.set_title("Overall accuracy", fontsize=12,
                 fontweight="bold", pad=12, color=TEXT_PRIMARY)

    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def main():
    data = load_analysis()

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), facecolor=SURFACE)
    fig.suptitle("Qwen2.5-1.5B: Base vs GRPO-Trained — 200 Question Eval",
                 fontsize=15, fontweight="bold", color=TEXT_PRIMARY, y=1.02)

    plot_per_type(axes[0], data["per_type"])
    plot_learned_forgotten(axes[1], data["learned_count"], data["forgotten_count"])
    plot_overall(axes[2], data["base_accuracy"], data["trained_accuracy"])

    fig.tight_layout()
    fig.savefig(OUTPUT_FILE, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    print(f"Plot saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
