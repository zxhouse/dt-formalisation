#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
Landscape-family robustness experiment  (reviewer request, Sec. 32-33).

The main paper characterises ONE utility landscape exactly.  A fair objection is
that parameter sweeps on a single landscape establish robustness to parameters,
not to the *class* of landscapes.  This script therefore generates a FAMILY of
independently drawn landscapes with randomised structure (feature count of the
needs, epistasis / pairwise-interaction strength, value spread), characterises
each by exhaustive enumeration (ruggedness = number of 1-flip local optima,
fitness-distance correlation FDC), and re-runs the Ideate-Prototype-Test loop on
every one.  It asks two questions the single-landscape study cannot:

  (Q1) Do the qualitative theorem signatures -- positive cumulative drift and
       convergence to a *local* optimum below U* -- hold ACROSS the family, or
       only on the tuned instance?
  (Q2) Is the headline ablation ordering "belief refinement (retest) matters
       more than ideation volume" a property of the algorithm, or an artefact of
       the one hand-built utility function?  We recompute both effects on every
       landscape and report the fraction of landscapes on which the ordering
       holds, with a landscape-level (random-effect) summary.

Dependencies: numpy, matplotlib, simulation.py.  Fixed seeds -> reproducible.
================================================================================
"""
import os
import json
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import simulation as S

OUTDIR = os.path.dirname(os.path.abspath(__file__))
OUT_JSON = os.path.join(OUTDIR, "landscapes_results.json")

# ---- experiment size ------------------------------------------------------ #
M_LANDSCAPES = 30         # number of independently drawn landscapes
K_RUNS       = 40         # design-process runs per landscape / per condition
N_ITER       = 70
RNG_SEED     = 20240619

mpl.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 320, "savefig.bbox": "tight",
    "font.family": "serif", "font.size": 13, "axes.titlesize": 14,
    "axes.titleweight": "bold", "axes.labelsize": 13,
    "xtick.labelsize": 11.5, "ytick.labelsize": 11.5, "legend.fontsize": 11,
    "axes.edgecolor": "#444444", "axes.linewidth": 0.9, "axes.grid": True,
    "grid.color": "#D5D8DC", "grid.linewidth": 0.7, "grid.alpha": 0.8,
    "mathtext.fontset": "cm",
})
C_PRIMARY, C_ACCENT, C_GREEN, C_GREY = "#2C5F8A", "#C0392B", "#1E8449", "#7F8C8D"


def make_landscape(seed, epistasis=1.0, n_needs=4, value_spread=1.0,
                   weights=(0.45, 0.30, 0.25)):
    """A UtilityLandscape whose ruggedness is modulated by `epistasis` (a scale
    on the pairwise interaction cost) and `value_spread` (a scale on per-feature
    business value), keeping U(p) in [0,1] and the additive D/V/F structure of
    the paper.  Larger epistasis -> more interactions -> more local optima."""
    land = S.UtilityLandscape(n_features=14, n_needs=n_needs, weights=weights,
                              seed=seed)
    land.inter = land.inter * float(epistasis)
    # keep viability meaningful: scale the value vector's spread about its mean
    v = land.value
    land.value = v.mean() + (v - v.mean()) * float(value_spread)
    return land


def draw_family(m=M_LANDSCAPES, seed=RNG_SEED):
    """Draw a structurally varied family of landscapes."""
    rng = np.random.default_rng(seed)
    fam = []
    for i in range(m):
        epi = float(rng.uniform(0.4, 3.0))          # epistasis / ruggedness knob
        nn = int(rng.integers(3, 7))                # 3..6 latent needs (modality)
        vs = float(rng.uniform(0.7, 1.6))           # value spread
        land = make_landscape(seed=10_000 + i, epistasis=epi, n_needs=nn,
                              value_spread=vs)
        fam.append({"idx": i, "epistasis": epi, "n_needs": nn,
                    "value_spread": vs, "land": land})
    return fam


def ensemble_finalX(land, gopt, k_runs=K_RUNS, **params):
    """Mean retained-best true utility X_inf and its across-run stats."""
    finals, drift_pos = [], 0
    for r in range(k_runs):
        out = S.run_design_process(land, n_iter=N_ITER, seed=3000 + r, **params)
        Xt = out["Xtrue"]
        finals.append(float(Xt[-1]))
        if Xt[-1] - Xt[0] > 0:
            drift_pos += 1
    finals = np.array(finals)
    return {
        "meanX": float(finals.mean()),
        "semX": float(finals.std(ddof=1) / np.sqrt(k_runs)),  # Monte-Carlo SE
        "gap": float(gopt - finals.mean()),
        "frac_local": float((finals < gopt - 1e-6).mean()),
        "frac_drift_pos": float(drift_pos / k_runs),
    }


def main():
    print("=" * 74)
    print("Landscape-family robustness experiment")
    print("=" * 74)
    fam = draw_family()
    rows = []
    for e in fam:
        land = e["land"]
        metr = S.characterize_landscape(land)
        gopt = metr["global_opt"]
        base = ensemble_finalX(land, gopt)                       # baseline
        noretest = ensemble_finalX(land, gopt, retest_top=1,
                                   explore_retest=False)         # impoverished
        moreidea = ensemble_finalX(land, gopt, ideas_per_round=8)  # more ideation
        # normalised performance = attained fraction of the global optimum
        row = {
            "idx": e["idx"], "epistasis": e["epistasis"], "n_needs": e["n_needs"],
            "value_spread": e["value_spread"],
            "n_local_optima": metr["n_local_optima"], "fdc": metr["fdc"],
            "global_opt": gopt, "mean_util": metr["mean_util"],
            "baseline_meanX": base["meanX"], "baseline_semX": base["semX"],
            "baseline_gap": base["gap"], "baseline_frac_local": base["frac_local"],
            "baseline_frac_drift_pos": base["frac_drift_pos"],
            "retest_benefit": base["meanX"] - noretest["meanX"],
            "ideation_benefit": moreidea["meanX"] - base["meanX"],
            "attained_frac": base["meanX"] / gopt,
        }
        rows.append(row)
        print(f"  L{e['idx']:02d}  optima={metr['n_local_optima']:3d} "
              f"fdc={metr['fdc']:+.2f}  X_inf={base['meanX']:.3f}"
              f"+/-{base['semX']:.3f}  drift+={base['frac_drift_pos']*100:3.0f}% "
              f"local={base['frac_local']*100:3.0f}%  "
              f"retest_gain={row['retest_benefit']:+.3f} "
              f"idea_gain={row['ideation_benefit']:+.3f}")

    rows_sorted = sorted(rows, key=lambda r: r["n_local_optima"])
    n_local = np.array([r["n_local_optima"] for r in rows])
    fdc = np.array([r["fdc"] for r in rows])
    gaps = np.array([r["baseline_gap"] for r in rows])
    attained = np.array([r["attained_frac"] for r in rows])
    drift_pos = np.array([r["baseline_frac_drift_pos"] for r in rows])
    frac_local = np.array([r["baseline_frac_local"] for r in rows])
    retest_b = np.array([r["retest_benefit"] for r in rows])
    idea_b = np.array([r["ideation_benefit"] for r in rows])

    # landscape-level (random-effect) summary: each landscape is one unit
    summary = {
        "M_landscapes": M_LANDSCAPES, "K_runs": K_RUNS, "N_iter": N_ITER,
        "n_local_optima": {"min": int(n_local.min()), "max": int(n_local.max()),
                            "mean": float(n_local.mean()),
                            "median": float(np.median(n_local))},
        "fdc": {"min": float(fdc.min()), "max": float(fdc.max()),
                "mean": float(fdc.mean())},
        # Q1: theorem signatures across the family
        "frac_landscapes_all_drift_pos": float((drift_pos > 0.999).mean()),
        "mean_frac_drift_pos": float(drift_pos.mean()),
        "min_frac_drift_pos": float(drift_pos.min()),
        "mean_frac_local": float(frac_local.mean()),
        "min_frac_local": float(frac_local.min()),
        "mean_attained_frac": float(attained.mean()),
        "attained_frac_range": [float(attained.min()), float(attained.max())],
        # relationship performance vs ruggedness (landscape as unit)
        "corr_gap_vs_localoptima": float(np.corrcoef(n_local, gaps)[0, 1]),
        "corr_gap_vs_fdc": float(np.corrcoef(fdc, gaps)[0, 1]),
        # Q2: ordering retest-benefit vs ideation-benefit across landscapes
        "mean_retest_benefit": float(retest_b.mean()),
        "mean_ideation_benefit": float(idea_b.mean()),
        "frac_landscapes_retest_gt_ideation":
            float((retest_b > idea_b).mean()),
        "retest_over_ideation_ratio":
            float(retest_b.mean() / max(1e-9, idea_b.mean())),
        "rows": rows_sorted,
    }
    with open(OUT_JSON, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n--- FAMILY SUMMARY ------------------------------------------------")
    print(f"  landscapes: M={M_LANDSCAPES}, runs/landscape/condition K={K_RUNS}")
    print(f"  local optima across family: {n_local.min()}..{n_local.max()} "
          f"(median {np.median(n_local):.0f});  FDC in "
          f"[{fdc.min():+.2f},{fdc.max():+.2f}]")
    print(f"  (Q1) positive cumulative drift in EVERY run on "
          f"{summary['frac_landscapes_all_drift_pos']*100:.0f}% of landscapes; "
          f"mean drift-positive fraction {summary['mean_frac_drift_pos']*100:.1f}%")
    print(f"       converged BELOW U* (local): mean "
          f"{summary['mean_frac_local']*100:.1f}% of runs "
          f"(min {summary['min_frac_local']*100:.0f}%)")
    print(f"       attained fraction of U*: mean "
          f"{summary['mean_attained_frac']*100:.1f}% "
          f"(range {attained.min()*100:.1f}-{attained.max()*100:.1f}%)")
    print(f"  gap vs #local-optima corr = "
          f"{summary['corr_gap_vs_localoptima']:+.2f}; "
          f"gap vs FDC corr = {summary['corr_gap_vs_fdc']:+.2f}")
    print(f"  (Q2) retest benefit {retest_b.mean():+.3f} vs ideation benefit "
          f"{idea_b.mean():+.3f}; retest>ideation on "
          f"{summary['frac_landscapes_retest_gt_ideation']*100:.0f}% "
          f"of landscapes")
    print(f"\nwritten -> {OUT_JSON}")

    make_figure(rows, summary)
    return summary


def make_figure(rows, summary):
    n_local = np.array([r["n_local_optima"] for r in rows])
    gaps = np.array([r["baseline_gap"] for r in rows])
    attained = np.array([r["attained_frac"] for r in rows])
    retest_b = np.array([r["retest_benefit"] for r in rows])
    idea_b = np.array([r["ideation_benefit"] for r in rows])

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))

    # (a) attained fraction of U* vs ruggedness -- theorem holds across family
    ax = axes[0]
    ax.scatter(n_local, attained * 100, s=42, color=C_PRIMARY, alpha=0.8,
               edgecolor="white", linewidth=0.6, zorder=3)
    ax.axhline(100, color=C_ACCENT, ls="--", lw=1.4,
               label="global optimum $U^*$")
    ax.set_xlabel("local optima (1-flip, exhaustive)")
    ax.set_ylabel(r"attained $\mathbb{E}[X_\infty]/U^*$  (\%)")
    ax.set_title("(a) Local convergence across the family")
    ax.legend(loc="lower left")

    # (b) gap vs FDC -- more rugged (FDC->0) => larger residual gap
    ax = axes[1]
    fdc = np.array([r["fdc"] for r in rows])
    ax.scatter(fdc, gaps, s=42, color=C_GREEN, alpha=0.8,
               edgecolor="white", linewidth=0.6, zorder=3)
    ax.set_xlabel("fitness-distance correlation (FDC)")
    ax.set_ylabel(r"residual gap $U^*-\mathbb{E}[X_\infty]$")
    ax.set_title("(b) Ruggedness sets the residual gap")

    # (c) per-landscape retest vs ideation benefit
    ax = axes[2]
    order = np.argsort(retest_b)
    x = np.arange(len(rows))
    ax.plot(x, retest_b[order], "o-", color=C_PRIMARY, ms=4, lw=1.3,
            label="belief-refinement (retest) benefit")
    ax.plot(x, idea_b[order], "s-", color=C_GREY, ms=3.5, lw=1.1,
            label="extra-ideation benefit")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xlabel("landscape (sorted)")
    ax.set_ylabel(r"$\Delta\,\mathbb{E}[X_\infty]$")
    ax.set_title("(c) Testing $>$ ideation on %.0f%% of landscapes"
                 % (summary["frac_landscapes_retest_gt_ideation"] * 100))
    ax.legend(loc="upper left")

    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, "Fig15.png"))
    print("wrote Fig15.png")


if __name__ == "__main__":
    main()
