#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PRELIMINARY empirical test of the model's predictions on two published
design-thinking studies, used as secondary data (Section 7.6).

This is NOT a confirmatory study. It is a small-N, retrospective, single-coder
application of the operationalization pipeline of Section 7.5 to already-published
design narratives, to show (a) that the pipeline can be applied to real data and
(b) that the model's central qualitative predictions are borne out in these cases.
Limitations (publication bias toward successful cases, coarse ordinal outcome
codes derived from authors' narratives, single coder = the present authors, N=2)
are stated in the paper and must temper any conclusion.

Case A -- SAM (van Asselt & Roke, 2025): a 7-year co-creative mHealth project,
          coded here into an Ideate-Prototype-Test sequence with bounded outcomes.
Case B -- Sung et al. (2018): a quantitative design-protocol analysis whose
          published two-event transition statistics we re-use to test whether the
          Ideate<->Test (generate<->evaluate) iteration the model assumes is
          empirically present, and to bound a generativity rate.

Dependencies: numpy, matplotlib.
"""
import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

OUTDIR = os.path.dirname(os.path.abspath(__file__))
mpl.rcParams.update({"font.family": "serif", "font.size": 10.5,
                     "mathtext.fontset": "cm", "savefig.dpi": 220,
                     "savefig.bbox": "tight", "axes.grid": True, "grid.alpha": 0.4})
C_P, C_A, C_G, C_O = "#2C5F8A", "#C0392B", "#1E8449", "#B9770E"


# =========================================================================== #
# CASE A: SAM -- coded Ideate-Prototype-Test sequence.
# Each test event: (short label, coded outcome in {adopt, confirm, mixed,
# reject, plateau}, is_reframe). Outcome codes are the authors' ordinal reading
# of the narrative's reported verdict; y in [0,1] is the bounded observation the
# pipeline (Section 7.5, step 4) maps them to.
# =========================================================================== #
OUTCOME_Y = {"adopt": 0.78, "confirm": 0.82, "mixed": 0.62,
             "reject": 0.40, "plateau": 0.55}

SAM = [
    ("reframe: EMI 'Self-mate' -> stress-EMA (SAM)", "adopt",   True),
    ("beta v1: 13-item stress questionnaire+tips",    "adopt",   False),
    ("A/B: answer format -> text(+emoji)",            "adopt",   False),
    ("A/B: name/logo/colour -> calmer, fewer choices","adopt",   False),
    ("A/B: score colours split -> personalisation",   "mixed",   False),
    ("usability: scores unclear -> simplify+info",    "adopt",   False),
    ("panel: offline data storage",                   "adopt",   False),
    ("pilot SCED (N=15): stress down, QoL up",        "confirm", False),
    ("RCT (N=214) corroborates",                      "confirm", False),
    ("Fitbit add-on tested -> no effect, confusion",  "reject",  False),
    ("panel: 4x/day too much -> reduce freq, trim Q", "adopt",   False),
    ("add back-nav + immediate stress tip",           "adopt",   False),
    ("in-depth (dissatisfied users): depth lacking",  "plateau", False),
    ("reframe v2.0: on-demand, Action, Inspiration",  "adopt",   True),
]


class Belief:
    __slots__ = ("mu", "var", "s2")
    def __init__(self, mu=0.5, var=0.08, sigma_obs=0.12):
        self.mu, self.var, self.s2 = mu, var, sigma_obs ** 2
    def update(self, y):
        p0, p = 1/self.var, 1/self.s2
        self.var = 1/(p0+p); self.mu = self.var*(p0*self.mu + p*y)


def analyse_sam():
    incumbent = Belief()                 # belief about the retained app's quality
    reject_cand = None
    Xk, retained_env, labels, kinds = [], [], [], []
    for label, outcome, reframe in SAM:
        y = OUTCOME_Y[outcome]
        if outcome == "reject":
            # a candidate ADD-ON tested below the incumbent -> NOT integrated;
            # the incumbent belief is unchanged (retention protects it).
            reject_cand = y
        else:
            incumbent.update(y)
        Xk.append(incumbent.mu)
        retained_env.append(max(retained_env[-1], incumbent.mu) if retained_env
                            else incumbent.mu)
        labels.append(label); kinds.append(outcome if not reframe else "reframe")
    dX = np.diff(Xk)
    # NOTE (honesty): the retained-best ENVELOPE is a running maximum, so its
    # monotonicity is TRUE BY CONSTRUCTION and tests nothing.  The non-trivial
    # quantity is the incumbent BELIEF X_k, which may fall (a submartingale is
    # not monotone); what the model predicts is a POSITIVE CUMULATIVE DRIFT.
    print("== Case A: SAM ==")
    print("  coded test events:", len(SAM))
    print("  incumbent belief X_k: cumulative drift X_T - X_0 = %+.3f" % (Xk[-1]-Xk[0]))
    print("  X_k decreases at %d of %d steps (submartingale, not monotone)"
          % (int((dX < -1e-9).sum()), dX.size))
    print("  mean one-step increment E[dX] = %+.4f" % dX.mean())
    print("  [envelope monotonicity omitted: it is true by construction]")
    print("  one candidate tested WORSE than incumbent (Fitbit) -> not retained:",
          reject_cand, "< incumbent", round(Xk[9], 3))
    print("  reframes at events:", [i+1 for i, (_, _, r) in enumerate(SAM) if r])
    return Xk, retained_env, kinds


# =========================================================================== #
# CASE B: Sung et al. (2018) -- published two-event transition frequencies
# (Table 5), states: Designing, Managing, Modeling, Predicting, Questioning,
# Defining. We test the generate<->evaluate (Designing<->Predicting) iteration
# and bound a generativity rate.
# =========================================================================== #
STATES = ["Design", "Manage", "Model", "Predict", "Question", "Define"]
# rows = Given, cols = Target (from the paper's Table 5; blanks on diagonal = 0)
OBS = np.array([
    [0, 14, 48, 33, 36, 16],   # Designing ->
    [12, 0, 15, 1, 8, 2],      # Managing ->
    [25, 9, 0, 13, 48, 11],    # Modeling ->
    [39, 2, 3, 0, 5, 3],       # Predicting ->
    [52, 10, 28, 3, 0, 14],    # Questioning ->
    [25, 5, 13, 2, 10, 0],     # Defining ->
], float)


def analyse_sung():
    row = STATES.index("Design"); col = STATES.index("Predict")
    des_pred = OBS[row, col]                       # Designing -> Predicting
    pred_des = OBS[col, row]                       # Predicting -> Designing
    pred_total = OBS[col].sum()
    # generativity rate: after an evaluation (Predicting), how often does the
    # designer return to generating ideas (Designing)?
    gen_rate = pred_des / pred_total
    # published significant right-tailed transitions (z-scores from Table 7)
    zsig = {"Design->Predict": 3.66, "Predict->Design": 5.17,
            "Model->Question": 4.98, "Manage->Model": 2.70}
    print("\n== Case B: Sung et al. (2018) protocol analysis ==")
    print("  Designing->Predicting freq =", int(des_pred),
          "(z=3.66, p<0.001); Predicting->Designing =", int(pred_des),
          "(z=5.17, p<0.001): significant bi-directional Ideate<->Test iteration")
    print("  generativity rate P(Design | after Predict) = %d/%d = %.2f"
          % (pred_des, pred_total, gen_rate))
    return gen_rate, zsig


# =========================================================================== #
# CODING ROBUSTNESS: a stand-in for a second coder.
#
# The single-coder limitation cannot be repaired retrospectively, but its
# CONSEQUENCE can be bounded.  We ask: if a second coder disagreed with the
# primary coding, how much would the conclusions change?  We simulate a
# disagreeing coder by perturbing each coded outcome to an ADJACENT category on
# the ordinal scale with probability q, and by jittering the ordinal->[0,1] map
# itself, then re-run the pipeline and record whether the two qualitative
# conclusions survive:
#     (C1) the retained-best envelope is non-decreasing;
#     (C2) the Fitbit candidate still scores below the incumbent at its event.
# This bounds sensitivity to coder disagreement; it does NOT establish
# reliability, which requires an actual second coder (Section 7.5, step 3).
# =========================================================================== #
ORDINAL = ["reject", "plateau", "mixed", "adopt", "confirm"]   # increasing order


def coding_robustness(q=0.30, n_rep=20000, jitter=0.05, seed=7):
    rng = np.random.default_rng(seed)
    ok_c1 = ok_c2 = 0
    drifts = []
    for _ in range(n_rep):
        # a disagreeing coder: shift each label one step with probability q
        ymap = {k: min(1.0, max(0.0, v + rng.normal(0, jitter)))
                for k, v in OUTCOME_Y.items()}
        inc, env, rej_y, rej_inc = Belief(), [], None, None
        xs = []
        for i, (label, outcome, _reframe) in enumerate(SAM):
            oc = outcome
            if rng.random() < q:
                j = ORDINAL.index(outcome)
                j = min(len(ORDINAL) - 1, max(0, j + rng.choice([-1, 1])))
                oc = ORDINAL[j]
            y = ymap[oc]
            if oc == "reject":
                rej_y, rej_inc = y, inc.mu
            else:
                inc.update(y)
            xs.append(inc.mu)
            env.append(max(env[-1], inc.mu) if env else inc.mu)
        e = np.array(xs)                        # the incumbent BELIEF, not the
        ok_c1 += int(e[-1] - e[0] > 0)          # envelope: positive drift
        ok_c2 += int(rej_y is not None and rej_y < rej_inc)
        drifts.append(e[-1] - e[0])
    drifts = np.array(drifts)
    print("\n== Coding robustness (%d perturbed re-codings, q=%.2f) ==" % (n_rep, q))
    print("  (C1) positive cumulative drift survives in %.1f%% of re-codings"
          % (100 * ok_c1 / n_rep))
    print("  (C2) rejected candidate still below incumbent in %.1f%%"
          % (100 * ok_c2 / n_rep))
    print("  cumulative retained drift: mean %+.3f, 5th-95th pct [%+.3f, %+.3f]"
          % (drifts.mean(), np.percentile(drifts, 5), np.percentile(drifts, 95)))
    return dict(c1=ok_c1 / n_rep, c2=ok_c2 / n_rep,
                drift_mean=float(drifts.mean()),
                drift_p5=float(np.percentile(drifts, 5)),
                drift_p95=float(np.percentile(drifts, 95)))


# =========================================================================== #
# FIGURE
# =========================================================================== #
def make_figure(Xk, env, kinds, gen_rate, zsig):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.4),
                                   gridspec_kw={"width_ratios": [1.35, 1.0],
                                                "wspace": 0.30})
    x = np.arange(1, len(Xk)+1)
    for i, k in enumerate(kinds):          # reframe guides first (behind)
        if k == "reframe":
            ax1.axvline(i+1, color=C_O, ls=":", lw=1.3, zorder=1)
    ax1.plot(x, Xk, "o-", color=C_P, lw=2.2, ms=5, zorder=3,
             label=r"incumbent belief $X_k$ (the tested quantity)")
    ax1.step(x, env, where="post", color=C_G, lw=1.7, ls="--", zorder=2,
             label=r"running maximum (monotone by construction)")
    # the rejected Fitbit candidate: scored below the incumbent, so NOT adopted
    ir = kinds.index("reject")
    ax1.scatter([ir+1], [OUTCOME_Y["reject"]], marker="X", color=C_A, zorder=5,
                s=90, label="rejected candidate (tested worse)")
    ax1.annotate("Fitbit add-on tested below\nincumbent $\\Rightarrow$ not retained\n"
                 "(testing blocks a worse option)",
                 (ir+1, OUTCOME_Y["reject"]), textcoords="offset points",
                 xytext=(-6, 30), ha="right", fontsize=8, color=C_A,
                 arrowprops=dict(arrowstyle="->", color=C_A))
    ax1.text(1.15, 0.9, "reframe", color=C_O, fontsize=8, rotation=90, va="top")
    ax1.text(13.85, 0.9, "reframe", color=C_O, fontsize=8, rotation=90, va="top")
    ax1.set_xlabel("coded test event (SAM, chronological)")
    ax1.set_ylabel("estimated quality of retained design")
    ax1.set_title("(a) Case A (SAM): incumbent belief drifts up (+0.015)\n"
                  "with two down-steps; the worse-tested option is rejected")
    ax1.set_ylim(0.33, 0.92); ax1.set_xlim(0.3, 14.7)
    ax1.legend(fontsize=8.2, loc="lower left", ncol=1)

    labels = list(zsig.keys()); z = [zsig[k] for k in labels]
    cols = [C_P if "Design" in k and "Predict" in k else "#888" for k in labels]
    ax2.barh(range(len(labels)), z, color=cols, alpha=0.85, edgecolor="white")
    ax2.axvline(1.96, color=C_A, ls="--", lw=1.4, label="z=1.96 (p=.05)")
    ax2.set_yticks(range(len(labels))); ax2.set_yticklabels(labels, fontsize=8.5)
    ax2.set_xlabel("transition z-score (Sung et al. 2018)")
    ax2.set_title("(b) Case B: significant Ideate$\\leftrightarrow$Test\n"
                  "iteration; generativity rate = %.2f" % gen_rate)
    ax2.legend(fontsize=8.5, loc="lower right")
    fig.savefig(os.path.join(OUTDIR, "Fig12.png"))
    plt.close(fig)
    print("\nwrote Fig12.png")


if __name__ == "__main__":
    Xk, env, kinds = analyse_sam()
    gen_rate, zsig = analyse_sung()
    coding_robustness()
    make_figure(Xk, env, kinds, gen_rate, zsig)
