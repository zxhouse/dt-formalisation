#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Figure 1 of the paper: the two levels of the model (frame and loop)."""
import os
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

HERE = os.path.dirname(os.path.abspath(__file__))
mpl.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm",
                     "savefig.dpi": 400, "savefig.bbox": "tight"})
INK, MUTED = "#1F2A37", "#5B6672"
BLUE, GREEN, AMBER = "#2C5F8A", "#1E8449", "#B9770E"
BAND_F, BAND_L = "#F3F6FA", "#F2F8F4"

fig, ax = plt.subplots(figsize=(7.2, 5.0))
ax.set_xlim(0, 100); ax.set_ylim(0, 70); ax.axis("off")


def band(y, h, face, edge, title, sub):
    ax.add_patch(FancyBboxPatch((1.5, y), 97, h, boxstyle="round,pad=0,rounding_size=2.2",
                                fc=face, ec=edge, lw=1.0, zorder=0))
    ax.text(4.5, y + h - 3.3, title, fontsize=11.5, color=edge, weight="bold", va="center")
    ax.text(4.5, y + h - 6.9, sub, fontsize=9.4, color=MUTED, va="center", style="italic")


def box(cx, cy, label, note, edge, w=19, h=8):
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                                boxstyle="round,pad=0,rounding_size=1.5",
                                fc="white", ec=edge, lw=1.6, zorder=3))
    ax.text(cx, cy, label, ha="center", va="center", fontsize=11.5, color=INK,
            weight="bold", zorder=4)
    ax.text(cx, cy - h / 2 - 1.5, note, ha="center", va="top", fontsize=9.0,
            color=MUTED, linespacing=1.3, zorder=4)


def arrow(p, q, color=INK, ls="-", rad=0.0, lw=1.5):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=14, lw=lw,
                                 color=color, linestyle=ls, zorder=2,
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0))


# ---- outer level: the frame ------------------------------------------------ #
band(45, 23.5, BAND_F, BLUE, "Choosing a frame",
     "the outer level: whose problem, and what counts as solving it")
box(36, 54.6, "Empathize", "observe users", BLUE)
box(64, 54.6, "Define", "commit to a point of view", BLUE)
arrow((45.5, 54.6), (54.5, 54.6))

# ---- inner level: the loop ------------------------------------------------- #
band(1.5, 36.5, BAND_L, GREEN, "The loop within a frame",
     "the inner level, repeated round after round; the theorem concerns this level")
xs = (22, 50, 78)
cy = 14.5
box(xs[0], cy, "Ideate", "propose variants of\nprototypes already built", GREEN)
box(xs[1], cy, "Prototype", "pre-screen the ideas\nand build one", GREEN)
box(xs[2], cy, "Test", "test, revise beliefs,\nchoose the favourite", GREEN)
arrow((31.5, cy), (40.5, cy))
arrow((59.5, cy), (68.5, cy))
arrow((xs[2] - 3, cy + 4), (xs[0] + 3, cy + 4), color=GREEN, rad=0.2)
ax.text(50, 26.4, "next round", ha="center", va="center", fontsize=9.4, color=GREEN,
        style="italic", zorder=5)

# ---- between the levels ---------------------------------------------------- #
arrow((40, 45), (40, 38), color=INK)
ax.text(38.3, 41.6, r"frame $(F,U)$", ha="right", va="center", fontsize=9.6, color=INK)
arrow((70, 38), (70, 45), color=AMBER, ls=(0, (4, 2.5)))
ax.text(71.7, 41.6, "reframe", ha="left", va="center", fontsize=9.6, color=AMBER, style="italic")

fig.savefig(os.path.join(HERE, "Fig1_v6.png"))
print("written Fig1_v6.png")
