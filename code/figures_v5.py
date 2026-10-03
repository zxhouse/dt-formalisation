#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figures of the v5 paper, drawn from results/*.json
(run studies.py first).

  FigA_v5.png  dynamics of the 80-round baseline
  FigB.png  long horizon: utility and exact local-optimality
  FigC.png  breadth vs depth of testing at a fixed budget
  FigD.png  loss caused by confirmation bias

Dependencies: numpy, matplotlib.
"""
import json
import os

import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")

mpl.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
    "font.family": "serif", "font.size": 9.5, "axes.titlesize": 10,
    "axes.labelsize": 9.5, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
    "legend.fontsize": 8.2, "axes.edgecolor": "#444444", "axes.linewidth": 0.8,
    "axes.grid": True, "grid.color": "#DDDDDD", "grid.linewidth": 0.6,
    "legend.frameon": False, "mathtext.fontset": "cm",
    "axes.spines.top": False, "axes.spines.right": False,
})
BLUE, GREEN, RED, ORANGE, GREY = "#2C5F8A", "#1E8449", "#B03A2E", "#B9770E", "#555555"


def load(name):
    return json.load(open(os.path.join(RES, name + "_results.json")))


def fig_dynamics():
    d = load("dynamics")
    u = d["U_star"]
    k = np.arange(1, d["rounds"] + 1)
    fig, ax = plt.subplots(figsize=(3.6, 2.75))
    spec = [("belief", "team's belief about its favourite", RED, "-"),
            ("found", "best prototype tested", GREEN, "-"),
            ("retained", "best favourite so far", ORANGE, "-"),
            ("shipped", "current favourite (shipped)", BLUE, "-")]
    for key, lab, col, ls in spec:
        m = np.array(d[key]["mean"]) / u * 100
        c = np.array(d[key]["ci95"]) / u * 100
        ax.plot(k, m, color=col, ls=ls, lw=1.5, label=lab)
        ax.fill_between(k, m - c, m + c, color=col, alpha=0.15, lw=0)
    ax.plot(k, np.array(d["calibrated_prior"]["belief_mean"]) / u * 100,
            color=RED, ls="--", lw=1.1, label="belief, calibrated prior")
    ax.axhline(100, color=GREY, lw=0.8, ls=":")
    ax.set_xlabel("round")
    ax.set_ylabel(r"utility, % of the optimum $U^*$")
    ax.set_ylim(86, 104)
    ax.legend(loc="lower right", handlelength=1.6, labelspacing=0.25)
    fig.savefig(os.path.join(HERE, "FigA_v5.png"))
    plt.close(fig)


def fig_horizon():
    d = load("horizon")
    sets = [("main_r2", "main landscape", BLUE), ("dvf_family_r2", "DVF family", GREEN),
            ("nk_family_r2", "NK family", RED)]
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.3, 2.75))
    for key, lab, col in sets:
        cp = d[key]["checkpoints"]
        x = np.array([int(c) for c in cp])
        a.plot(x, [100 * cp[str(c)]["found"] for c in x], color=col, ls="--", marker="o", ms=3, lw=1.2)
        a.plot(x, [100 * cp[str(c)]["shipped"] for c in x], color=col, ls="-", marker="o", ms=3, lw=1.5, label=lab)
        b.plot(x, [100 * cp[str(c)]["best_tested_is_local_opt"] for c in x], color=col, ls="--", marker="o", ms=3, lw=1.2)
        b.plot(x, [100 * cp[str(c)]["incumbent_is_local_opt"] for c in x], color=col, ls="-", marker="o", ms=3, lw=1.5, label=lab)
    for ax in (a, b):
        ax.set_xscale("log", base=2)
        ax.set_xticks([5, 10, 20, 40, 80, 160, 320, 640, 1280, 2560])
        ax.set_xticklabels(["5", "10", "20", "40", "80", "160", "320", "640", "1280", "2560"], fontsize=7.5)
        ax.set_xlabel("rounds")
    a.axhline(100, color=GREY, lw=0.8, ls=":")
    a.set_ylabel(r"true utility, % of $U^*$")
    a.set_title("(a) Shipped (solid) and best tested (dashed)")
    a.legend(loc="lower right")
    b.set_ylabel("runs at a local optimum, %")
    b.set_title("(b) Local optimality, checked exactly")
    b.set_ylim(-3, 103)
    b.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "FigB_v5.png"))
    plt.close(fig)


def fig_allocation():
    d = load("allocation")
    cols = {"0.025": GREEN, "0.05": BLUE, "0.1": ORANGE, "0.2": RED}
    names = [("main", "(a) Main landscape"), ("dvf", "(b) DVF family"), ("nk", "(c) NK family")]
    fig, axes = plt.subplots(1, 3, figsize=(7.3, 2.6), sharex=True)
    for ax, (key, title) in zip(axes, names):
        for sg, col in cols.items():
            cell = d["cells"][key][f"sigma={sg},ideas=5"]["by_n_new"]
            arms = d["arms"]
            x = np.array([float(a) for a in arms])
            m = np.array([cell[a]["mean"] for a in arms]) * 100
            c = np.array([cell[a]["ci95"] for a in arms]) * 100
            ax.errorbar(x, m, yerr=c, color=col, marker="o", ms=3, lw=1.4, capsize=2,
                        label=rf"$\sigma={float(sg):g}$")
            j = int(np.argmax(m))
            ax.plot(x[j], m[j], marker="o", ms=7.5, mfc="none", mec=col, mew=1.2)
        ax.set_title(title)
        ax.set_xticks([0.5, 1, 2, 3, 4, 5])
        ax.set_xticklabels(["½", "1", "2", "3", "4", "5"])
    fig.supxlabel("new prototypes per round (out of 5 tests; the rest are re-tests of the three leading prototypes)", fontsize=9.5, y=0.04)
    axes[0].set_ylabel(r"shipped utility, % of $U^*$")
    axes[2].legend(loc="lower left", title="test noise", title_fontsize=8.2)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "FigC_v5.png"))
    plt.close(fig)


def fig_bias():
    d = load("bias")["by_sigma"]
    fig, axes = plt.subplots(1, 3, figsize=(7.3, 2.5), sharey=True)
    for ax, sg in zip(axes, ("0.05", "0.1", "0.2")):
        for key, lab, col in [("main", "main landscape", BLUE), ("dvf", "DVF family", GREEN),
                              ("nk", "NK family", RED)]:
            b = sorted(d[sg][key], key=float)
            x = np.array([float(v) for v in b])
            m = np.array([d[sg][key][v]["shipped"] for v in b]) * 100
            c = np.array([d[sg][key][v]["shipped_ci95"] for v in b]) * 100
            ax.errorbar(x, m - m[0], yerr=c, color=col, marker="o", ms=3, lw=1.3, capsize=2, label=lab)
        ax.axvline(float(sg), color=GREY, lw=0.8, ls="--")
        ax.text(float(sg) + 0.006, -13.2, r"$b=\sigma$", fontsize=8.5, color=GREY)
        ax.set_title(rf"test noise $\sigma={float(sg):g}$")
        ax.set_ylim(-14.5, 2.5)
        ax.set_xticks([0, 0.1, 0.2, 0.3])
    axes[0].set_ylabel(r"change in shipped utility, % of $U^*$")
    axes[2].legend(loc="lower left", bbox_to_anchor=(0.0, 0.1))
    fig.supxlabel(r"confirmation bias $b$ added to every re-test of the favourite", fontsize=9.5, y=0.04)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "FigD_v5.png"))
    plt.close(fig)


if __name__ == "__main__":
    fig_dynamics(); fig_horizon(); fig_allocation(); fig_bias()
    print("written FigA_v5-FigD_v5")
