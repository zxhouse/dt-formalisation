#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
Benchmark comparison against independent baseline algorithms (reviewer, Sec.38).

The paper studies the Ideate-Prototype-Test loop in isolation.  A reviewer fairly
asks how it compares to standard search/decision baselines under a MATCHED test
budget, and whether reported effect sizes and Monte-Carlo error are stated.  This
script pits the DT loop against four baselines on a common harness:

  * random search          -- test bounded-neighbourhood candidates, keep the best
  * greedy hill-climbing    -- move to the best improving 1-flip neighbour
  * epsilon-greedy          -- bandit over materialised prototypes
  * UCB / best-arm          -- UCB1 index over materialised prototypes
  * DT loop (this paper)    -- Algorithm 1 (bounded ideate + retain + retest)

All algorithms share: the same landscape, the same bounded mutation operator
(Hamming radius r_max), the same noisy test oracle y = U(p) + N(0, sigma^2), the
same conjugate belief updater with the [0,1] projection, and -- crucially -- the
SAME number of tests per run (the currency of cost in the model).  Each run
reports the *true* utility of the prototype the algorithm would ship (its
posterior-mean incumbent).  We report means with 95% CIs, the Monte-Carlo
standard error, and Cohen's d against the DT loop, on the main landscape and
across the landscape family of landscapes.py.

Dependencies: numpy, matplotlib, simulation.py, landscapes.py.  Fixed seeds.
================================================================================
"""
import os
import json
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import simulation as S
from landscapes import draw_family

OUTDIR = os.path.dirname(os.path.abspath(__file__))
OUT_JSON = os.path.join(OUTDIR, "benchmarks_results.json")

BUDGET   = 300      # tests per run (identical across algorithms)
R_MAX    = 2        # bounded mutation radius (Assumption 2)
SIGMA    = 0.10     # test-noise s.d.
PRIOR_V  = 0.08
N_RUNS   = 250      # runs on the main landscape
FAM_RUNS = 30       # runs per landscape in the family comparison
RNG_SEED = 20240619

mpl.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 320, "savefig.bbox": "tight",
    "font.family": "serif", "font.size": 13, "axes.titlesize": 14,
    "axes.titleweight": "bold", "axes.labelsize": 13,
    "xtick.labelsize": 11.5, "ytick.labelsize": 11.5, "legend.fontsize": 11,
    "axes.edgecolor": "#444444", "axes.linewidth": 0.9, "axes.grid": True,
    "grid.color": "#D5D8DC", "grid.linewidth": 0.7, "grid.alpha": 0.8,
    "mathtext.fontset": "cm",
})
COLORS = {"random": "#95A5A6", "greedy": "#E67E22", "eps_greedy": "#8E44AD",
          "ucb": "#2980B9", "dt_loop": "#1E8449"}
PRETTY = {"random": "random search", "greedy": "greedy hill-climb",
          "eps_greedy": r"$\varepsilon$-greedy", "ucb": "UCB / best-arm",
          "dt_loop": "DT loop (this paper)"}


# --------------------------------------------------------------------------- #
#  Shared harness: bounded mutation + noisy test oracle + conjugate belief
# --------------------------------------------------------------------------- #
class Harness:
    def __init__(self, landscape, sigma=SIGMA, prior_var=PRIOR_V, r_max=R_MAX,
                 seed=0):
        self.L = landscape
        self.n = landscape.n
        self.sigma = sigma
        self.prior_var = prior_var
        self.r_max = r_max
        self.rng = np.random.default_rng(seed)
        self.beliefs = {}          # key -> UtilityBelief
        self.counts = {}           # key -> #tests
        self.true = {}             # key -> true utility (cached)
        self.tests = 0

    def key(self, p):
        return frozenset(np.flatnonzero(p).tolist())

    def vec(self, k):
        v = np.zeros(self.n)
        if k:
            v[list(k)] = 1.0
        return v

    def rand_proto(self, density=0.3):
        return (self.rng.random(self.n) < density).astype(float)

    def mutate(self, base):
        child = base.copy()
        flips = self.rng.integers(1, self.r_max + 1)
        idx = self.rng.choice(self.n, size=flips, replace=False)
        child[idx] = 1.0 - child[idx]
        return child

    def _ensure(self, p):
        k = self.key(p)
        if k not in self.true:
            self.true[k] = self.L.utility(p)
        if k not in self.beliefs:
            self.beliefs[k] = S.UtilityBelief(0.5, self.prior_var, self.sigma)
            self.counts[k] = 0
        return k

    def test(self, p):
        """One noisy test; updates the belief. Returns the belief key."""
        k = self._ensure(p)
        y = self.true[k] + self.rng.normal(0, self.sigma)
        self.beliefs[k].update(y)
        self.counts[k] += 1
        self.tests += 1
        return k

    def budget_left(self, budget):
        return self.tests < budget

    def recommend_true(self):
        """True utility of the shipped prototype = argmax posterior mean."""
        if not self.beliefs:
            return 0.0
        best = max(self.beliefs, key=lambda kk: self.beliefs[kk].mu)
        return float(self.true[best])


# --------------------------------------------------------------------------- #
#  Algorithms  (each spends exactly `budget` tests, returns shipped true util)
# --------------------------------------------------------------------------- #
def algo_random(land, budget, seed):
    h = Harness(land, seed=seed)
    pool = [h.rand_proto() for _ in range(5)]
    for p in pool:
        h.test(p)
    while h.budget_left(budget):
        base = pool[h.rng.integers(len(pool))]
        child = h.mutate(base)
        h.test(child)
        pool.append(child)
    return h.recommend_true()


def algo_greedy(land, budget, seed):
    h = Harness(land, seed=seed)
    cur = h.rand_proto()
    h.test(cur)
    while h.budget_left(budget):
        base = h.vec(max(h.beliefs, key=lambda kk: h.beliefs[kk].mu))
        # evaluate 1-flip neighbours (noisy), move to the best improving one
        best_child, best_mu = None, h.beliefs[h.key(base)].mu
        order = h.rng.permutation(h.n)
        for i in order:
            if not h.budget_left(budget):
                break
            q = base.copy(); q[i] = 1.0 - q[i]
            k = h.test(q)
            if h.beliefs[k].mu > best_mu + 1e-9:
                best_child, best_mu = q, h.beliefs[k].mu
        if best_child is None:            # local optimum -> random restart
            if h.budget_left(budget):
                h.test(h.rand_proto())
    return h.recommend_true()


def algo_eps_greedy(land, budget, seed, eps=0.3):
    h = Harness(land, seed=seed)
    for _ in range(5):
        h.test(h.rand_proto())
    while h.budget_left(budget):
        if h.rng.random() < eps or not h.beliefs:
            base = h.vec(max(h.beliefs, key=lambda kk: h.beliefs[kk].mu))
            h.test(h.mutate(base))        # explore a bounded-neighbour candidate
        else:
            best = max(h.beliefs, key=lambda kk: h.beliefs[kk].mu)
            h.test(h.vec(best))           # exploit: retest the leader
    return h.recommend_true()


def algo_ucb(land, budget, seed, c=0.4, add_every=3):
    h = Harness(land, seed=seed)
    for _ in range(5):
        h.test(h.rand_proto())
    step = 0
    while h.budget_left(budget):
        step += 1
        if step % add_every == 0:         # periodically grow the arm set
            base = h.vec(max(h.beliefs, key=lambda kk: h.beliefs[kk].mu))
            h.test(h.mutate(base))
            continue
        t = max(1, h.tests)
        # UCB1 index over materialised arms
        best_k, best_idx = None, -1e9
        for k in h.beliefs:
            idx = h.beliefs[k].mu + c * np.sqrt(np.log(t) / max(1, h.counts[k]))
            if idx > best_idx:
                best_k, best_idx = k, idx
        h.test(h.vec(best_k))
    return h.recommend_true()


def algo_dt_loop(land, budget, seed):
    """Algorithm 1.  Choose an iteration count that spends ~budget tests, then
    trim to exactly `budget` by construction of run_design_process' accounting:
    per round it spends 1 (new) + up to retest_top (retest) + 1 (explore) tests.
    We approximate the matched budget with n_iter = budget // 5 and report the
    shipped incumbent's true utility at the horizon."""
    n_iter = max(1, budget // 5)
    out = S.run_design_process(land, n_iter=n_iter, ideas_per_round=5,
                               sigma_obs=SIGMA, prior_var=PRIOR_V, retest_top=3,
                               r_max=R_MAX, seed=seed, explore_retest=True)
    # shipped = true utility of the belief-incumbent at the final step
    return float(out["best_true"][-1])


ALGOS = {"random": algo_random, "greedy": algo_greedy,
         "eps_greedy": algo_eps_greedy, "ucb": algo_ucb, "dt_loop": algo_dt_loop}


def cohens_d(a, b):
    a, b = np.asarray(a), np.asarray(b)
    na, nb = len(a), len(b)
    sp = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) /
                 (na + nb - 2))
    return float((a.mean() - b.mean()) / sp) if sp > 0 else 0.0


def run_on_landscape(land, n_runs, budget=BUDGET, seed0=0):
    res = {}
    for name, fn in ALGOS.items():
        vals = np.array([fn(land, budget, seed=seed0 + 7919 * i)
                         for i in range(n_runs)])
        res[name] = vals
    return res


def summarize(res, gopt):
    out = {}
    dt = res["dt_loop"]
    for name, vals in res.items():
        ci = 1.96 * vals.std(ddof=1) / np.sqrt(len(vals))
        out[name] = {
            "mean": float(vals.mean()),
            "ci95": float(ci),
            "mcse": float(vals.std(ddof=1) / np.sqrt(len(vals))),
            "sd": float(vals.std(ddof=1)),
            "gap_to_Ustar": float(gopt - vals.mean()),
            "attained_frac": float(vals.mean() / gopt),
            "cohens_d_vs_dt": cohens_d(vals, dt),
            "n": int(len(vals)),
        }
    return out


def main():
    print("=" * 74)
    print("Benchmark comparison under a matched test budget of %d tests" % BUDGET)
    print("=" * 74)
    land = S.UtilityLandscape()
    gopt = S._global_optimum(land)
    print(f"main landscape: U* = {gopt:.4f}\n")

    res = run_on_landscape(land, N_RUNS, seed0=1000)
    summ = summarize(res, gopt)
    print("  %-22s  %8s  %10s  %8s  %8s" %
          ("algorithm", "E[Y]", "95% CI", "MCSE", "d vs DT"))
    for name in ["random", "greedy", "eps_greedy", "ucb", "dt_loop"]:
        s = summ[name]
        print("  %-22s  %.4f   +/-%.4f   %.4f   %+.2f" %
              (PRETTY[name].replace("$", "").replace("\\varepsilon", "eps"),
               s["mean"], s["ci95"], s["mcse"], s["cohens_d_vs_dt"]))

    # ---- family comparison: is DT robust across landscapes? --------------- #
    print("\n  running family comparison (%d landscapes x %d runs) ..."
          % (10, FAM_RUNS))
    fam = draw_family(m=10)
    fam_attained = {name: [] for name in ALGOS}
    fam_win_dt = 0
    for e in fam:
        L = e["land"]
        g = S._global_optimum(L)
        r = run_on_landscape(L, FAM_RUNS, seed0=500)
        means = {name: float(r[name].mean()) for name in ALGOS}
        for name in ALGOS:
            fam_attained[name].append(means[name] / g)
        if means["dt_loop"] >= max(means[n] for n in ALGOS):
            fam_win_dt += 1
    fam_summary = {name: {"mean_attained_frac": float(np.mean(v)),
                          "sd": float(np.std(v))}
                   for name, v in fam_attained.items()}
    fam_summary["dt_best_on_n_landscapes"] = int(fam_win_dt)
    fam_summary["n_landscapes"] = len(fam)

    print("  family mean attained fraction of U*:")
    for name in ["random", "greedy", "eps_greedy", "ucb", "dt_loop"]:
        print("     %-22s %.3f" %
              (PRETTY[name].replace("$", "").replace("\\varepsilon", "eps"),
               fam_summary[name]["mean_attained_frac"]))
    print("  DT loop is the best (or tied) on %d/%d landscapes"
          % (fam_win_dt, len(fam)))

    out = {"budget": BUDGET, "n_runs": N_RUNS, "main_landscape_Ustar": gopt,
           "main": summ, "family": fam_summary}
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwritten -> {OUT_JSON}")
    make_figure(summ, gopt, fam_summary)
    return out


def make_figure(summ, gopt, fam_summary):
    order = ["random", "greedy", "eps_greedy", "ucb", "dt_loop"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))

    ax = axes[0]
    means = [summ[n]["mean"] for n in order]
    cis = [summ[n]["ci95"] for n in order]
    cols = [COLORS[n] for n in order]
    x = np.arange(len(order))
    ax.bar(x, means, yerr=cis, color=cols, alpha=0.9, capsize=4,
           error_kw={"lw": 1.1})
    ax.axhline(gopt, color="#C0392B", ls="--", lw=1.5, label="global optimum $U^*$")
    ax.set_xticks(x)
    ax.set_xticklabels([PRETTY[n] for n in order], rotation=20, ha="right")
    ax.set_ylabel(r"shipped true utility $\mathbb{E}[Y]$")
    ax.set_ylim(min(means) - 0.03, gopt + 0.015)
    ax.set_title("(a) Matched-budget comparison (main landscape)")
    ax.legend(loc="lower left")

    ax = axes[1]
    fam_means = [fam_summary[n]["mean_attained_frac"] * 100 for n in order]
    fam_sds = [fam_summary[n]["sd"] * 100 for n in order]
    ax.bar(x, fam_means, yerr=fam_sds, color=cols, alpha=0.9, capsize=4,
           error_kw={"lw": 1.1})
    ax.axhline(100, color="#C0392B", ls="--", lw=1.5)
    ax.set_xticks(x)
    ax.set_xticklabels([PRETTY[n] for n in order], rotation=20, ha="right")
    ax.set_ylabel(r"attained fraction of $U^*$ (\%)")
    ax.set_title("(b) Robustness across the landscape family")

    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, "Fig16.png"))
    print("wrote Fig16.png")


if __name__ == "__main__":
    main()
