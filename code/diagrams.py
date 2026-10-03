#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Regenerates the two schematic diagrams of the paper as clean, publication-quality
figures (consistent with the simulation figures' serif style):

  Fig2.png : the Design Thinking process as a directed state-space graph G=(V,A).
  Fig1.png : the DT process flow with backtracking / iteration loops.

Pure matplotlib (no graphviz). Color choices are also distinguishable in
grayscale (shape + position carry the information), per Design Studies artwork
accessibility guidance.
"""
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUTDIR = os.path.dirname(os.path.abspath(__file__))

C_NODE   = "#EAF1F8"   # light blue fill
C_NODE_E = "#2C5F8A"   # node edge / divergent
C_CONV   = "#C0392B"   # convergent accent
C_DIV    = "#1E8449"   # divergent accent
C_TEXT   = "#1B2631"
C_BACK   = "#B9770E"   # backtracking arrows

mpl.rcParams.update({
    "font.family": "serif", "font.size": 11, "mathtext.fontset": "cm",
    "savefig.dpi": 300, "savefig.bbox": "tight",
})


def _box(ax, xy, w, h, text, fc=C_NODE, ec=C_NODE_E, fs=10.5, tc=C_TEXT,
         lw=1.6, boxstyle="round,pad=0.02,rounding_size=0.10"):
    x, y = xy
    ax.add_patch(FancyBboxPatch((x - w/2, y - h/2), w, h, boxstyle=boxstyle,
                                linewidth=lw, edgecolor=ec, facecolor=fc,
                                mutation_scale=1, zorder=3))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, color=tc,
            zorder=4, linespacing=1.25)


def _arrow(ax, p0, p1, color=C_NODE_E, lw=1.8, style="-|>", rad=0.0,
           ls="-", z=2):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=15,
                                 lw=lw, color=color, ls=ls,
                                 connectionstyle=f"arc3,rad={rad}", zorder=z,
                                 shrinkA=2, shrinkB=2))


def _elabel(ax, x, y, text, color, fs=9.5, rot=0, ha="center"):
    ax.text(x, y, text, ha=ha, va="center", fontsize=fs, color=color,
            rotation=rot, zorder=5,
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none",
                      alpha=0.9))


# =========================================================================== #
#  Fig2 : directed state-space graph  G=(V,A)
# =========================================================================== #
def fig_state_graph(fname):
    fig, ax = plt.subplots(figsize=(9.0, 8.8))
    ax.set_xlim(-1.2, 12.0); ax.set_ylim(0.6, 13.6); ax.axis("off")

    sx = 4.0  # spine x
    ys = {"v0": 12.6, "v1": 10.6, "v2": 8.6, "v3": 6.6, "v4": 4.6, "v5": 2.6}
    P = {k: (sx, y) for k, y in ys.items()}
    labels = {
        "v0": r"$v_0$: initial state" + "\n" + r"$(I_0,N_0,P_{v,0},S_{i,0},P_{r,0})$",
        "v1": r"$v_1$: after Empathize" + "\n" + r"insight set $I$ expanded",
        "v2": r"$v_2$: after Define" + "\n" + r"point of view $pv^*$ fixed",
        "v3": r"$v_3$: after Ideate" + "\n" + r"idea set $S_i$ expanded",
        "v4": r"$v_4$: after Prototype" + "\n" + r"$p^*$ added to $P_r$",
        "v5": r"$v_5$: after Test" + "\n" + r"belief updated (Bayes)",
    }
    w, h = 4.2, 1.1
    for k, xy in P.items():
        _box(ax, xy, w, h, labels[k], fs=9.4)

    ops = [
        ("v0", "v1", r"$Op_{\mathrm{Empathize}}$", C_DIV),
        ("v1", "v2", r"$Op_{\mathrm{Define}}$", C_CONV),
        ("v2", "v3", r"$Op_{\mathrm{Ideate}}$", C_DIV),
        ("v3", "v4", r"$Op_{\mathrm{Prototype}}$", C_CONV),
        ("v4", "v5", r"$Op_{\mathrm{Test}}$", C_CONV),
    ]
    for a, b, lab, col in ops:
        x0, y0 = P[a]; x1, y1 = P[b]
        _arrow(ax, (x0, y0 - h/2), (x1, y1 + h/2), color=col, lw=2.0)
        _elabel(ax, sx + w/2 - 0.35, (y0 + y1)/2, lab, col, fs=9, ha="left")

    def cloop(y_from, y_to, x_edge, x_rail, label):
        """Orthogonal C-shaped routing: out to a rail, along it, back in."""
        ax.plot([x_edge, x_rail], [y_from, y_from], color=C_BACK, lw=2.0,
                ls=(0, (5, 2)), zorder=1)
        ax.plot([x_rail, x_rail], [y_from, y_to], color=C_BACK, lw=2.0,
                ls=(0, (5, 2)), zorder=1)
        _arrow(ax, (x_rail, y_to), (x_edge, y_to), color=C_BACK, lw=2.0,
               ls=(0, (5, 2)), z=1)
        side = 0.45 if x_rail > x_edge else -0.45
        _elabel(ax, x_rail + side, (y_from + y_to) / 2, label, C_BACK, fs=8.4,
                rot=90, ha="center")

    # iteration loop on the right: Test -> Ideate (the I-P-T cycle)
    cloop(ys["v5"], ys["v3"], sx + w/2, 9.4,
          "iterate (Ideate--Prototype--Test)")
    # backtracking loop on the left: Test -> Empathize (reframe)
    cloop(ys["v5"], ys["v1"], sx - w/2, -0.7, "re-empathize (reframe)")

    handles = [
        plt.Line2D([0], [0], color=C_DIV, lw=2.4, label="divergent operator"),
        plt.Line2D([0], [0], color=C_CONV, lw=2.4, label="convergent operator"),
        plt.Line2D([0], [0], color=C_BACK, lw=2.0, ls="--",
                   label="iteration / backtracking"),
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=8.7,
              framealpha=0.95, borderpad=0.6, bbox_to_anchor=(1.0, 1.0))
    ax.set_title(r"Design Thinking as a stochastic walk on the state-space "
                 r"graph $G=(V,A)$", fontsize=11.5, fontweight="bold", pad=4)
    fig.savefig(fname)
    plt.close(fig)


# =========================================================================== #
#  Fig1 : process flow with backtracking / iteration loops
# =========================================================================== #
def fig_process_flow(fname):
    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.set_xlim(0, 22); ax.set_ylim(0, 9); ax.axis("off")

    phases = [
        ("Empathize", r"$\uparrow|I|$", C_DIV),
        ("Define",    r"$pv^*$",        C_CONV),
        ("Ideate",    r"$\uparrow|S_i|$", C_DIV),
        ("Prototype", r"$p^*\!\in\!P_r$", C_CONV),
        ("Test",      r"$P(p\mid\mathcal{E})$", C_CONV),
    ]
    xs = [2.2, 6.0, 9.8, 13.6, 17.4]
    y = 6.2
    w, h = 3.0, 1.7
    centers = {}
    for (name, sub, col), x in zip(phases, xs):
        _box(ax, (x, y), w, h, "", ec=col, lw=2.2)
        ax.text(x, y + 0.28, name, ha="center", va="center", fontsize=12.5,
                fontweight="bold", color=C_TEXT, zorder=5)
        ax.text(x, y - 0.45, sub, ha="center", va="center", fontsize=11,
                color=col, zorder=5)
        centers[name] = x
        # phase type tag
        tag = "divergent" if col == C_DIV else "convergent"
        ax.text(x, y + h/2 + 0.32, tag, ha="center", fontsize=8.5,
                style="italic", color=col)

    # forward arrows
    for i in range(len(xs) - 1):
        _arrow(ax, (xs[i] + w/2, y), (xs[i+1] - w/2, y), color="#34495E", lw=2.2)

    # backtracking arcs (below): Test->Ideate, Test->Empathize, Prototype->Ideate, Define->Empathize
    backs = [
        ("Test", "Ideate", "test reveals a weak concept", -0.32, 3.7),
        ("Test", "Empathize", "test reveals misread needs", -0.5, 2.2),
        ("Prototype", "Ideate", "infeasible to build", -0.28, 4.6),
        ("Define", "Empathize", "frame too narrow", -0.34, 4.6),
    ]
    for a, b, lab, rad, ytxt in backs:
        xa, xb = centers[a], centers[b]
        _arrow(ax, (xa, y - h/2), (xb, y - h/2), color=C_BACK, lw=1.6,
               rad=rad, ls=(0, (5, 2)))
        _elabel(ax, (xa + xb)/2, ytxt, lab, C_BACK, fs=8.3)

    # divergence/convergence band annotation
    ax.annotate("", xy=(xs[-1] + 0.2, 8.4), xytext=(xs[0] - 0.2, 8.4),
                arrowprops=dict(arrowstyle="-", color="#999999", lw=1))
    ax.text(xs[0] - 0.2, 8.6, "problem space  +  solution space  (iterated)",
            fontsize=9, color="#666666")

    handles = [
        plt.Line2D([0], [0], color="#34495E", lw=2.4, label="forward transition"),
        plt.Line2D([0], [0], color=C_BACK, lw=1.8, ls="--",
                   label="backtracking / iteration loop"),
        plt.Line2D([0], [0], marker="s", color="w", markerfacecolor="w",
                   markeredgecolor=C_DIV, markersize=11,
                   label="divergent phase ($\\uparrow$ entropy)"),
        plt.Line2D([0], [0], marker="s", color="w", markerfacecolor="w",
                   markeredgecolor=C_CONV, markersize=11,
                   label="convergent phase ($\\downarrow$ entropy)"),
    ]
    ax.legend(handles=handles, loc="lower center", ncol=2, fontsize=9,
              framealpha=0.95, bbox_to_anchor=(0.5, -0.02))
    ax.set_title("Process flow of Design Thinking with backtracking and "
                 "iteration loops", fontsize=12, fontweight="bold", pad=6)
    fig.savefig(fname)
    plt.close(fig)


if __name__ == "__main__":
    fig_state_graph(os.path.join(OUTDIR, "Fig2.png"))
    fig_process_flow(os.path.join(OUTDIR, "Fig1.png"))
    print("wrote Fig2.png (state-space graph) and Fig1.png (process flow)")
