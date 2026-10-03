#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ILLUSTRATIVE proof-of-concept for the operationalization pipeline of Section 7.5.

This is NOT an empirical validation. It applies the coding-and-estimation pipeline
to a small, hand-authored *synthetic* design protocol so that the abstract steps
(segment -> code test events -> estimate bounded posteriors -> check the retained-
best submartingale) become concrete and executable. Real validation requires real
protocol data; here we only demonstrate that the pipeline is well defined and that,
on a protocol satisfying the paper's assumptions, the predicted submartingale
behaviour is recovered from coded observations rather than assumed.

Output: Fig11.png and a short printed report.
"""
import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

OUTDIR = os.path.dirname(os.path.abspath(__file__))
mpl.rcParams.update({"font.family": "serif", "font.size": 11,
                     "mathtext.fontset": "cm", "savefig.dpi": 220,
                     "savefig.bbox": "tight", "axes.grid": True,
                     "grid.alpha": 0.5})
C_P, C_A, C_G = "#2C5F8A", "#C0392B", "#1E8449"

# --------------------------------------------------------------------------- #
# 1. A synthetic, hand-coded protocol (stands in for a coded transcript).
#    Each entry: (segment label, coded phase, concept id, 5-point test outcome).
#    Phases: I = Ideate, P = Prototype, T = Test. Test outcomes are Likert 1..5
#    usability/desirability ratings that a second coder would also assign
#    (inter-rater reliability is reported for real data; here outcomes are given).
# --------------------------------------------------------------------------- #
PROTOCOL = [
    ("\"what if the cup folds flat?\"",              "I", "A", None),
    ("built foam model A; 5 users try it",            "PT", "A", 3),   # ok-ish
    ("\"add a leak-proof lid to A\"",                 "I", "B", None),
    ("built B; 5 users try it",                       "PT", "B", 4),   # better
    ("\"try a rigid version instead\"",               "I", "C", None),
    ("built rigid C; users find it bulky",            "PT", "C", 3),   # worse
    ("re-test B with a new group",                    "PT", "B", 4),   # confirm B
    ("\"B plus a heat sleeve\"",                      "I", "D", None),
    ("built D; users like it a lot",                  "PT", "D", 5),   # best
    ("re-test D to be sure",                          "PT", "D", 4),   # confirm D
    ("\"D but cheaper materials\"",                   "I", "E", None),
    ("built E; slightly less liked",                  "PT", "E", 4),   # tie-ish
]


class Belief:
    """Normal-normal bounded belief about a concept's quality q in [0,1]."""
    def __init__(self, mu=0.5, var=0.08, sigma_obs=0.12):
        self.mu, self.var, self.s2 = mu, var, sigma_obs ** 2

    def update(self, y):
        p0, p = 1 / self.var, 1 / self.s2
        self.var = 1 / (p0 + p)
        self.mu = self.var * (p0 * self.mu + p * y)


def likert_to_obs(s):
    """Map a 1..5 Likert rating to a bounded quality observation y in [0,1]."""
    return (s - 1) / 4.0


def run_pipeline(protocol):
    beliefs, order = {}, []
    means_over_time = {}          # concept -> list of (step, posterior mean)
    Xk = []                       # belief X_k = max posterior mean (submartingale)
    incumbents = []               # argmax-belief incumbent identity per step
    test_steps, test_labels = [], []
    step = 0
    for label, phase, cid, outcome in protocol:
        if "T" in phase and outcome is not None:
            step += 1
            if cid not in beliefs:
                beliefs[cid] = Belief()
                order.append(cid)
            beliefs[cid].update(likert_to_obs(outcome))
            Xk.append(max(b.mu for b in beliefs.values()))
            incumbents.append(max(beliefs, key=lambda c: beliefs[c].mu))
            test_steps.append(step)
            test_labels.append(f"{cid}:{outcome}")
            for c in order:
                means_over_time.setdefault(c, []).append((step, beliefs[c].mu))
    envelope = list(np.maximum.accumulate(Xk))   # running max (retained best)
    return (beliefs, order, means_over_time, Xk, envelope, incumbents,
            test_steps, test_labels)


def main():
    (beliefs, order, means_over_time, Xk, env, incs,
     steps, labels) = run_pipeline(PROTOCOL)

    dX = np.diff(Xk)
    nonneg = float(np.mean(dX >= -1e-9))
    print("Illustrative coded protocol (synthetic; NOT empirical validation)")
    print("  tested concepts:", order)
    print("  belief X_k trajectory :", [round(v, 3) for v in Xk])
    print("  retained-best envelope:", [round(v, 3) for v in env])
    print("  incumbent identity    :", incs)
    print(f"  X_k non-decreasing at {nonneg*100:.0f}% of coded steps; "
          f"the one dip is a re-test that corrected an over-estimate")
    print(f"  cumulative drift X_T-X_0 = {Xk[-1]-Xk[0]:+.3f} (>0 as predicted); "
          f"incumbent never switched to a worse concept")

    # ---- figure ---- #
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.3),
                                   gridspec_kw={"width_ratios": [1.1, 1.0],
                                                "wspace": 0.28})
    cmap = plt.cm.tab10(np.linspace(0, 1, 10))
    for i, c in enumerate(order):
        xs = [s for s, _ in means_over_time[c]]
        ys = [m for _, m in means_over_time[c]]
        ax1.plot(xs, ys, "o-", color=cmap[i], lw=1.8, ms=5, label=f"concept {c}")
    ax1.set_xlabel("test event (coded from protocol)")
    ax1.set_ylabel("posterior mean quality  $\\mathbb{E}[q\\mid$ evidence$]$")
    ax1.set_title("(a) Per-concept beliefs, estimated from coded ratings")
    ax1.legend(fontsize=8, ncol=2)

    ax2.plot(steps, Xk, "o-", color=C_P, lw=2.2, ms=5,
             label=r"belief $X_k=\max_c\mathbb{E}[q_c]$")
    ax2.step(steps, env, where="post", color=C_G, lw=1.8, ls="--",
             label=r"retained-best envelope $\max_{j\leq k}X_j$")
    for s, lab in zip(steps, labels):
        ax2.annotate(lab, (s, Xk[s - 1]), textcoords="offset points",
                     xytext=(0, 7), ha="center", fontsize=7.5, color="#555")
    ax2.annotate("re-test corrects\nan over-estimate", (6, Xk[5]),
                 textcoords="offset points", xytext=(-2, -34), ha="center",
                 fontsize=8, color=C_A,
                 arrowprops=dict(arrowstyle="->", color=C_A))
    ax2.set_xlabel("test event")
    ax2.set_ylabel(r"estimated quality")
    ax2.set_title(r"(b) $X_k$ has positive drift (submartingale)")
    ax2.set_ylim(0.45, 0.98)
    ax2.legend(fontsize=8.5, loc="lower right")
    fig.savefig(os.path.join(OUTDIR, "Fig11.png"))
    plt.close(fig)
    print("wrote Fig11.png")


if __name__ == "__main__":
    main()
