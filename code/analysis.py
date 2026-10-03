#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extended quantitative analysis for the Design-Thinking convergence model.

Produces, for the article's Results section:
  * a parameter-sensitivity sweep (observation noise, ideas-per-round, retest);
  * an ablation study isolating the role of each modelling assumption;
  * the distribution (percentiles) of the limit utility X_inf;
  * a statistical (confidence-interval) test of the submartingale drift.

Runs are resumable: results accumulate in `analysis_results.json`, so the script
can be invoked repeatedly (it skips configurations already computed) and respects
a soft wall-clock budget per invocation (env MAXTIME, default 38 s).

Dependencies: numpy (+ simulation.py in the same directory).
"""
import os, json, time
import numpy as np
import simulation as S

OUTDIR = os.path.dirname(os.path.abspath(__file__))
JSON = os.path.join(OUTDIR, "analysis_results.json")
R_RUNS = 120          # ensemble size per configuration
N_ITER = 70
EPS, NW = 0.01, 5     # epsilon-stopping params
MAXTIME = float(os.environ.get("MAXTIME", "38"))

# Baseline operator parameters (match simulation.run_design_process defaults)
# Baseline: BOUNDED mutation neighborhood (r_max=2), no global restart,
# unbiased testing.  This is the regime in which "local optimum" is the true
# almost-sure limit (reviewer R3.2 fix).
BASE = dict(ideas_per_round=5, sigma_obs=0.10, retest_top=3,
            p_restart=0.0, r_max=2, test_bias=0.0)

# ---- configurations ------------------------------------------------------- #
def cfg(name, group, **over):
    d = dict(BASE); d.update(over)
    return {"name": name, "group": group, "params": d}

CONFIGS = [
    # sensitivity: observation noise
    cfg("sigma=0.05", "noise", sigma_obs=0.05),
    cfg("sigma=0.10*", "noise", sigma_obs=0.10),
    cfg("sigma=0.20", "noise", sigma_obs=0.20),
    cfg("sigma=0.30", "noise", sigma_obs=0.30),
    # sensitivity: ideas per round (divergence strength)
    cfg("ideas=2", "ideas", ideas_per_round=2),
    cfg("ideas=5*", "ideas", ideas_per_round=5),
    cfg("ideas=8", "ideas", ideas_per_round=8),
    # sensitivity: retest budget (belief refinement)
    cfg("retest=1", "retest", retest_top=1),
    cfg("retest=3*", "retest", retest_top=3),
    cfg("retest=6", "retest", retest_top=6),
    # sensitivity: confirmation bias magnitude (Assumption 3 violation)
    cfg("bias=0.00*", "bias", test_bias=0.0),
    cfg("bias=0.05", "bias", test_bias=0.05),
    cfg("bias=0.10", "bias", test_bias=0.10),
    cfg("bias=0.20", "bias", test_bias=0.20),
    # sensitivity: neighborhood radius r_max
    cfg("rmax=1", "nbhd", r_max=1),
    cfg("rmax=2*", "nbhd", r_max=2),
    cfg("rmax=3", "nbhd", r_max=3),
    # sensitivity: prior variance (belief informativeness)
    cfg("prior=0.03", "prior", prior_var=0.03),
    cfg("prior=0.08*", "prior", prior_var=0.08),
    cfg("prior=0.20", "prior", prior_var=0.20),
    # ablations vs baseline
    cfg("baseline", "ablation"),
    cfg("confirmation bias", "ablation", test_bias=0.10),
    cfg("no retest (greedy belief)", "ablation", retest_top=0),
    cfg("global restart (unbounded)", "ablation", p_restart=0.4),
]


def evaluate(params, gopt):
    """Run an ensemble and return summary metrics."""
    runs = [S.run_design_process(LAND, n_iter=N_ITER, seed=2000 + r, **params)
            for r in range(R_RUNS)]
    Xt = np.vstack([r["Xtrue"] for r in runs])         # (R, N_ITER)
    finals = Xt[:, -1]
    # per-run convergence iteration
    kstar = []
    for r in range(R_RUNS):
        marg = Xt[r, NW:] - Xt[r, :-NW]
        below = np.where(marg < EPS)[0]
        kstar.append(int(below[0]) if len(below) else N_ITER - NW)
    kstar = np.array(kstar)
    # submartingale drift with 95% CI lower bound
    incr = np.diff(Xt, axis=0 if False else 1)         # (R, N_ITER-1)
    dmean = incr.mean(0)
    dsem = incr.std(0) / np.sqrt(R_RUNS)
    ci_low = dmean - 1.96 * dsem
    # joint (multiplicity-safe) test of the drift: the cumulative increment
    # Xtrue[-1]-Xtrue[0] is a single statistic; report its mean and 95% CI.
    cum = Xt[:, -1] - Xt[:, 0]
    cum_ci = 1.96 * cum.std() / np.sqrt(R_RUNS)
    return {
        "finalX": float(finals.mean()),
        "finalX_ci95": float(1.96 * finals.std() / np.sqrt(R_RUNS)),
        "sdX": float(finals.std()),
        "gap": float(gopt - finals.mean()),
        "frac_local": float(np.mean(finals < gopt - 1e-6)),
        "kstar_median": float(np.median(kstar)),
        "kstar_p90": float(np.percentile(kstar, 90)),
        "drift_nonneg_frac": float(np.mean(dmean >= -1e-9)),
        "drift_min_ci": float(ci_low.min()),
        "cum_drift": float(cum.mean()),
        "cum_drift_ci95": float(cum_ci),
        "pctl": [float(np.percentile(finals, q)) for q in (5, 25, 50, 75, 95)],
    }


def main():
    global LAND
    LAND = S.UtilityLandscape()
    gopt = S._global_optimum(LAND)

    results = {}
    if os.path.exists(JSON):
        results = json.load(open(JSON))
    land_metrics = S.characterize_landscape(LAND)
    results["_meta"] = {"global_opt": float(gopt), "R_RUNS": R_RUNS,
                        "N_ITER": N_ITER, "eps": EPS, "N": NW,
                        "landscape": land_metrics}
    print("  landscape:", land_metrics)

    t0 = time.time()
    done_now = 0
    for c in CONFIGS:
        # recompute if missing OR if stored params differ from the current config
        old = results.get(c["name"])
        if old is not None and old.get("params") == c["params"]:
            continue
        if time.time() - t0 > MAXTIME and done_now > 0:
            print("time budget reached; re-run to continue.")
            break
        ts = time.time()
        results[c["name"]] = {"group": c["group"], "params": c["params"],
                              **evaluate(c["params"], gopt)}
        json.dump(results, open(JSON, "w"), indent=1)
        done_now += 1
        print(f"  [{c['group']:9s}] {c['name']:30s} "
              f"finalX={results[c['name']]['finalX']:.4f} "
              f"local={results[c['name']]['frac_local']*100:4.0f}% "
              f"k*med={results[c['name']]['kstar_median']:.0f} "
              f"({time.time()-ts:.1f}s)")

    done = [c["name"] for c in CONFIGS
            if results.get(c["name"], {}).get("params") == c["params"]]
    print(f"\nconfigs complete: {len(done)}/{len(CONFIGS)}  "
          f"(global_opt={gopt:.4f})")
    if len(done) == len(CONFIGS):
        make_outputs(results, gopt)
        print("ALL DONE -> Fig10.png, sensitivity_rows.tex, ablation_rows.tex")


def make_outputs(res, gopt):
    """Build the sensitivity figure and LaTeX table rows from the JSON."""
    import matplotlib.pyplot as plt
    S  # ensure style applied via simulation import
    C1, C2, C3 = "#2C5F8A", "#C0392B", "#1E8449"

    import matplotlib as _mpl
    _mpl.rcParams.update({"font.size": 14, "axes.titlesize": 15,
                          "axes.labelsize": 14, "xtick.labelsize": 12.5,
                          "ytick.labelsize": 13, "legend.fontsize": 12.5})
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.7))

    # (a) final utility & gap vs observation noise
    noise = ["sigma=0.05", "sigma=0.10*", "sigma=0.20", "sigma=0.30"]
    xs = [0.05, 0.10, 0.20, 0.30]
    fx = [res[n]["finalX"] for n in noise]
    gp = [res[n]["gap"] for n in noise]
    ax = axes[0]
    ax.plot(xs, fx, "o-", color=C1, lw=2.2, label=r"$\mathbb{E}[X_\infty]$")
    ax.axhline(gopt, color=C2, ls="--", lw=1.4, label=r"$U^*$")
    ax.set_xlabel(r"Observation noise $\sigma$")
    ax.set_ylabel(r"Final utility $\mathbb{E}[X_\infty]$")
    ax.set_title("(a) Robustness to test noise")
    ax.legend(fontsize=9)

    # (b) effect of CONFIRMATION BIAS (Assumption 3 violation)
    bias = ["bias=0.00*", "bias=0.05", "bias=0.10", "bias=0.20"]
    bx = [0.0, 0.05, 0.10, 0.20]
    fb = [res[n]["finalX"] for n in bias]
    eb = [res[n]["finalX_ci95"] for n in bias]
    ax2 = axes[1]
    ax2.errorbar(bx, fb, yerr=eb, fmt="s-", color=C2, lw=2.2, capsize=3)
    ax2.set_xlabel(r"Confirmation bias $b$")
    ax2.set_ylabel(r"Final utility $\mathbb{E}[X_\infty]$", color=C2)
    ax2.tick_params(axis="y", labelcolor=C2)
    ax2.set_title("(b) Effect of biased testing")

    # (c) ablation bars
    abl = ["baseline", "confirmation bias",
           "no retest (greedy belief)", "global restart (unbounded)"]
    labs = ["baseline", "confirm.\nbias", "no\nretest", "global\nrestart"]
    vals = [res[a]["finalX"] for a in abl]
    errs = [res[a]["finalX_ci95"] for a in abl]
    ax3 = axes[2]
    bars = ax3.bar(range(len(abl)), vals, yerr=errs, capsize=3,
                   color=[C1, C2, C2, C3], alpha=0.85, edgecolor="white")
    ax3.axhline(gopt, color="#555", ls="--", lw=1.2, label=r"$U^*$")
    ax3.set_xticks(range(len(abl)))
    ax3.set_xticklabels(labs, fontsize=8)
    ax3.set_ylabel(r"Final utility $\mathbb{E}[X_\infty]$")
    ax3.set_ylim(0.69, gopt + 0.015)
    ax3.set_title("(c) Ablation of assumptions")
    for b, v in zip(bars, vals):
        ax3.text(b.get_x() + b.get_width()/2, v + errs[0] + 0.002,
                 f"{v:.3f}", ha="center", fontsize=8)
    ax3.legend(fontsize=9, loc="lower left")

    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, "Fig10.png"), dpi=320, bbox_inches="tight")
    plt.close(fig)

    # LaTeX rows: sensitivity table (finalX with +/- 95% CI)
    sens_order = ["sigma=0.05", "sigma=0.10*", "sigma=0.20", "sigma=0.30",
                  "ideas=2", "ideas=5*", "ideas=8",
                  "retest=1", "retest=3*", "retest=6",
                  "bias=0.00*", "bias=0.05", "bias=0.10", "bias=0.20",
                  "rmax=1", "rmax=2*", "rmax=3",
                  "prior=0.03", "prior=0.08*", "prior=0.20"]
    def row(n):
        r = res[n]
        nm = (n.replace("*", "$^\\dagger$").replace("sigma", "$\\sigma$")
               .replace("rmax", "$r_{\\max}$").replace("bias", "$b$")
               .replace("prior", "$\\sigma_0^2$"))
        return (f"{nm} & {r['finalX']:.4f}\\,$\\pm$\\,{r['finalX_ci95']:.4f} & "
                f"{r['gap']:.4f} & {r['frac_local']*100:.0f}\\% & "
                f"{r['kstar_median']:.0f} & {r['drift_nonneg_frac']*100:.0f}\\% \\\\")
    open(os.path.join(OUTDIR, "sensitivity_rows.tex"), "w").write(
        "\n".join(row(n) for n in sens_order) + "\n")

    abl_order = ["baseline", "confirmation bias",
                 "no retest (greedy belief)", "global restart (unbounded)"]
    def arow(n):
        r = res[n]
        return (f"{n} & {r['finalX']:.4f}\\,$\\pm$\\,{r['finalX_ci95']:.4f} & "
                f"{r['gap']:.4f} & {r['frac_local']*100:.0f}\\% & "
                f"{r['kstar_median']:.0f} \\\\")
    open(os.path.join(OUTDIR, "ablation_rows.tex"), "w").write(
        "\n".join(arow(n) for n in abl_order) + "\n")

    # percentiles + joint drift test for the baseline
    b = res["sigma=0.10*"]
    print("\nBaseline X_inf percentiles [p5,p25,p50,p75,p95] =",
          [round(v, 4) for v in b["pctl"]])
    print(f"Baseline joint drift test: cumulative E[X_T-X_0] = "
          f"{b['cum_drift']:.4f} +/- {b['cum_drift_ci95']:.4f} (95% CI)")
    print("Landscape:", res["_meta"]["landscape"])
    print("LaTeX rows written.")


if __name__ == "__main__":
    main()
