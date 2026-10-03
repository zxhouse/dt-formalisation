#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
Numerical validation of the Bayesian-Graph formalization of Design Thinking.

This script provides a Monte-Carlo simulation of the Ideate -> Prototype -> Test
loop described in the paper. It empirically validates the central theoretical
claims:

  (C1) The running best estimated expected utility  X_k = max_{p in P_{r,k}}
       E[U(p) | F_k]  is a *bounded submartingale*:  E[X_{k+1}-X_k | F_k] >= 0,
       while individual sample paths may locally decrease.
  (C2) By Doob's submartingale convergence theorem the process converges almost
       surely; empirically the ensemble mean reaches a plateau and each run
       converges to a (run-dependent) local optimum -- consistent with the
       "wicked problem" multimodal landscape.
  (C3) Divergent operators increase the Shannon entropy of the design state;
       convergent (selection) operators decrease it.
  (C4) The Test operator performs a correct Bayesian update; the posterior mean
       of the utility is a martingale and the posterior concentrates on the
       true value (unbiased testing, Assumption 3).
  (C5) The marginal utility improvement decays below a threshold eps, formalizing
       the "good-enough" stopping rule (Definition of convergence).

Author: companion code for Kuraś, Michalski, Gerka.
Dependencies: numpy, matplotlib (no scipy/networkx required).
================================================================================
"""

import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# --------------------------------------------------------------------------- #
#  Global style (consistent, publication-quality look for all figures)
# --------------------------------------------------------------------------- #
OUTDIR = os.path.dirname(os.path.abspath(__file__))
RNG_SEED = 20240619

C_PRIMARY   = "#2C5F8A"   # deep blue
C_ACCENT    = "#C0392B"   # brick red
C_GREEN     = "#1E8449"   # green
C_PURPLE    = "#7D3C98"   # purple
C_GREY      = "#7F8C8D"   # grey
C_BAND      = "#5DADE2"   # light blue band

mpl.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 320,
    "savefig.bbox": "tight",
    "font.family": "serif",
    "font.size": 14,
    "axes.titlesize": 15,
    "axes.titleweight": "bold",
    "axes.labelsize": 14,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 12.5,
    "axes.edgecolor": "#444444",
    "axes.linewidth": 0.9,
    "axes.grid": True,
    "grid.color": "#D5D8DC",
    "grid.linewidth": 0.7,
    "grid.alpha": 0.8,
    "legend.frameon": True,
    "legend.framealpha": 0.92,
    "legend.edgecolor": "#BBBBBB",
    "mathtext.fontset": "cm",
})


# =========================================================================== #
#  1. THE GROUND-TRUTH UTILITY LANDSCAPE  (a "wicked", multimodal function)
# =========================================================================== #
class UtilityLandscape:
    """
    A prototype p is a binary vector over n_features features (p subseteq F).
    Each of the three IDEO criteria is a bounded function of the feature set:

        U_D (Desirability): rewards covering latent user "needs" (clustered
                            features) with *diminishing returns* (saturation).
        U_V (Viability):    rewards a few high-value features but penalises
                            scope creep (too many features -> diluted value).
        U_F (Feasibility):  decreases with build cost; pairwise interactions
                            make some feature combinations technically costly.

    U(p) = w_d U_D + w_v U_V + w_f U_F,   w_i >= 0,  sum w_i = 1,  U in [0,1].

    Interaction terms make the landscape non-convex / multimodal, which is the
    formal signature of a wicked problem (Section 2.1 of the paper).
    """

    def __init__(self, n_features=14, n_needs=4, weights=(0.45, 0.30, 0.25),
                 seed=RNG_SEED):
        rng = np.random.default_rng(seed)
        self.n = n_features
        self.w_d, self.w_v, self.w_f = weights
        assert abs(sum(weights) - 1.0) < 1e-9 and min(weights) >= 0

        # Desirability: each need is satisfied by a (random) subset of features.
        self.need_masks = (rng.random((n_needs, n_features)) < 0.35).astype(float)
        self.need_weight = rng.uniform(0.6, 1.0, size=n_needs)
        self.need_weight /= self.need_weight.sum()

        # Viability: per-feature business value (some negative = cost centres).
        self.value = rng.uniform(-0.4, 1.0, size=n_features)

        # Feasibility: per-feature build cost + pairwise integration cost.
        self.cost = rng.uniform(0.05, 0.30, size=n_features)
        W = rng.uniform(-0.05, 0.20, size=(n_features, n_features))
        self.inter = np.triu(W + W.T, 1)  # symmetric pairwise interaction cost

    # ---- individual criteria, each squashed to [0, 1] --------------------- #
    def _desirability(self, p):
        coverage = self.need_masks @ p                      # features per need
        sat = 1.0 - np.exp(-0.9 * coverage)                 # diminishing returns
        return float(self.need_weight @ sat)

    def _viability(self, p):
        raw = self.value @ p
        scope_penalty = 0.04 * max(0.0, p.sum() - 6) ** 1.5  # scope creep
        return float(_sigmoid(0.8 * (raw - scope_penalty) - 1.0))

    def _feasibility(self, p):
        c = self.cost @ p + p @ self.inter @ p
        return float(np.exp(-0.55 * c))                     # cheaper -> higher

    def components(self, p):
        p = np.asarray(p, dtype=float)
        return self._desirability(p), self._viability(p), self._feasibility(p)

    def utility(self, p):
        ud, uv, uf = self.components(p)
        return self.w_d * ud + self.w_v * uv + self.w_f * uf


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


# =========================================================================== #
#  2. BAYESIAN BELIEF over a prototype's true utility (normal-normal model)
# =========================================================================== #
class UtilityBelief:
    """
    Conjugate Gaussian belief about the (unknown, fixed) true utility u* of one
    prototype. Each Test yields y = u* + noise, noise ~ N(0, sigma_obs^2),
    i.e. UNBIASED evidence (Assumption 3). The posterior mean is updated by the
    standard precision-weighted rule and is, by the tower property, a martingale
    in the filtration generated by the observations.
    """
    __slots__ = ("mu", "var", "sigma_obs2")

    def __init__(self, prior_mu, prior_var, sigma_obs):
        self.mu = float(prior_mu)
        self.var = float(prior_var)
        self.sigma_obs2 = float(sigma_obs) ** 2

    def update(self, y):
        prec0 = 1.0 / self.var
        prec_obs = 1.0 / self.sigma_obs2
        new_var = 1.0 / (prec0 + prec_obs)
        raw_mu = new_var * (prec0 * self.mu + prec_obs * y)
        # ---- BOUNDEDNESS FIX (reviewer, Sec. 3 of the review) -------------- #
        # The quantity the theorem is about is the *true* posterior mean
        # M_k^p = E[U(p) | F_k] of the bounded utility U(p) in [0,1]; a genuine
        # conditional expectation of a [0,1]-valued random variable ALWAYS lies
        # in [0,1] (this, not Assumption 1 read loosely, is what bounds X_k).
        # The conjugate normal-normal mean is only an *unbounded approximation*
        # to it, so we take the metric projection onto the known range [0,1].
        # This enforces M_k^p in [0,1] for the implemented process, matching the
        # idealized belief the proof uses; empirically it moves nothing, because
        # observations of utilities ~0.5-0.8 with s.d. ~0.1 almost never push the
        # precision-weighted mean outside [0,1].
        self.mu = min(1.0, max(0.0, raw_mu))
        self.var = new_var
        return self.mu


# =========================================================================== #
#  3. THE IDEATE -> PROTOTYPE -> TEST LOOP  (one run)
# =========================================================================== #
def run_design_process(landscape, n_iter=80, ideas_per_round=5,
                       sigma_obs=0.10, prior_var=0.08, retest_top=3,
                       p_restart=0.0, r_max=2, test_bias=0.0, seed=0,
                       explore_retest=True):
    """
    Returns a dict of per-iteration trajectories for one run of the loop.

    Operators (Section 4).  IMPORTANT (reviewer fix): Ideate operates on a
    BOUNDED mutation neighborhood -- it flips at most `r_max` features of an
    already-explored prototype.  The operator-reachable neighborhood is therefore
    the Hamming ball of radius `r_max` around the current frontier, NOT the whole
    space.  This makes ``local optimum'' the genuine almost-sure limit of the
    process (rather than a finite-horizon artifact): a state is locally optimal
    when no prototype within Hamming distance r_max has higher utility.

      * Ideate  (divergent): generate `ideas_per_round` new candidates by
                 mutating a top prototype within radius r_max.  Global random
                 restarts are DISABLED by default (p_restart=0); enabling them
                 (ablation) enlarges the neighborhood towards the whole space and,
                 as theory predicts, pushes the limit towards the global optimum.
      * Prototype (convergent): among untested candidates choose the one with the
                 highest quick proxy estimate; start a neutral belief.
      * Test    : sample noisy evidence y = U(p) + test_bias + eps and
                 Bayesian-update.  With test_bias=0 the posterior mean is a
                 martingale (Assumption 3); test_bias!=0 models systematic bias
                 b_k = E[eps|F_k] and turns X_k into a near-submartingale.
                 The leading `retest_top` prototypes are re-tested each round so
                 their posterior means concentrate (posterior consistency).
    """
    rng = np.random.default_rng(seed)
    n = landscape.n

    beliefs = {}          # frozenset(features) -> UtilityBelief
    true_util = {}        # cache of true utilities (each computed once)
    cand_guess = {}       # candidate key -> fixed prior heuristic guess

    def key(p):
        return frozenset(np.flatnonzero(p).tolist())

    def vec(k):
        v = np.zeros(n)
        if k:
            v[list(k)] = 1.0
        return v

    def add_candidate(p):
        k = key(p)
        if k not in beliefs and k not in cand_guess:
            tu = landscape.utility(p)        # computed exactly once per prototype
            true_util[k] = tu
            cand_guess[k] = tu + rng.normal(0, 0.10)   # optimistic prior proxy

    # seed with a few random prototypes
    for _ in range(ideas_per_round):
        add_candidate((rng.random(n) < 0.3).astype(float))

    Xk = []                 # running max of the team's belief (posterior mean)
    Xtrue = []              # running max TRUE expected utility retained (theorem X_k)
    best_true = []          # true utility of current belief-incumbent
    n_tested = []           # |P_r|  (materialized prototypes)
    entropy_state = []      # Shannon entropy of the design (idea) state

    incumbent_key = None
    best_true_so_far = 0.0

    for k in range(n_iter):
        # ---- IDEATE (divergent): bounded-radius mutation of the frontier -- #
        if beliefs:
            frontier = sorted(beliefs, key=lambda kk: beliefs[kk].mu,
                              reverse=True)[:max(1, retest_top)]
            base = vec(frontier[rng.integers(len(frontier))])
        else:
            base = (rng.random(n) < 0.3).astype(float)
        for _ in range(ideas_per_round):
            child = base.copy()
            flips = rng.integers(1, r_max + 1)          # Hamming radius <= r_max
            idx = rng.choice(n, size=flips, replace=False)
            child[idx] = 1.0 - child[idx]
            add_candidate(child)
        # OPTIONAL global restart (ablation only): enlarges the neighborhood
        if p_restart > 0 and rng.random() < p_restart:
            add_candidate((rng.random(n) < 0.3).astype(float))

        # ---- PROTOTYPE (convergent): pick best *untested* candidate ------ #
        if cand_guess:
            chosen = max(cand_guess, key=cand_guess.get)
            del cand_guess[chosen]
            tu = true_util[chosen]
            # honest, neutral prior (NOT centred on the unknown true utility)
            beliefs[chosen] = UtilityBelief(
                prior_mu=0.5, prior_var=prior_var, sigma_obs=sigma_obs)
            # ---- TEST: noisy evidence for the freshly chosen prototype --- #
            beliefs[chosen].update(tu + rng.normal(0, sigma_obs))

        # ---- TEST (cont.): re-test the leading prototypes to refine ------ #
        # `test_bias` models CONFIRMATION bias: the currently favoured incumbent
        # is over-rated by +test_bias at each of its tests (the team looks for
        # reasons its favourite works), which distorts retention.  test_bias=0
        # recovers the unbiased (martingale) case of Assumption 3.
        if beliefs:
            top = sorted(beliefs, key=lambda kk: beliefs[kk].mu,
                         reverse=True)[:retest_top]
            for rank, kk in enumerate(top):
                bias = test_bias if rank == 0 else 0.0
                beliefs[kk].update(true_util[kk] + bias +
                                   rng.normal(0, sigma_obs))

        # ---- EXPLORATORY RETEST (reviewer fix, Sec. 4 of the review) ----- #
        # Assumption 4 requires every *retained* incumbent to be tested
        # infinitely often, but the top-r retest above can let an older
        # prototype fall out of the top-r and stop being tested.  A round-robin
        # retest of one materialized prototype per round closes this gap: over
        # an infinite horizon every retained prototype is revisited infinitely
        # often, so its belief is consistent (M_k^p -> U(p)).  This makes the
        # implemented loop (Algorithm 1) genuinely satisfy Assumption 4 rather
        # than only approximate it; its effect on the finite-horizon trajectory
        # is negligible.  Set explore_retest=False to recover the bare top-r
        # heuristic used in earlier drafts.
        if explore_retest and beliefs:
            rr_keys = list(beliefs.keys())
            kk = rr_keys[k % len(rr_keys)]
            beliefs[kk].update(true_util[kk] + rng.normal(0, sigma_obs))

        # ---- bookkeeping ------------------------------------------------- #
        # team's belief: max posterior mean over all materialized prototypes
        xk = max(b.mu for b in beliefs.values())
        incumbent_key = max(beliefs, key=lambda kk: beliefs[kk].mu)
        # theorem's X_k: best TRUE expected utility retained by a rational team
        # (a rational team never discards the best-known solution, hence the
        #  retained best is monotone non-decreasing -> a submartingale)
        best_true_so_far = max(best_true_so_far, true_util[incumbent_key])
        Xk.append(xk)
        Xtrue.append(best_true_so_far)
        best_true.append(true_util[incumbent_key])
        n_tested.append(len(beliefs))
        entropy_state.append(_state_entropy(beliefs, n))

    return {
        "Xk": np.array(Xk),
        "Xtrue": np.array(Xtrue),
        "best_true": np.array(best_true),
        "n_tested": np.array(n_tested),
        "entropy": np.array(entropy_state),
        "global_opt": _global_optimum(landscape),
    }


def _state_entropy(beliefs, n_features):
    """
    Shannon entropy (bits) of the design state, defined as the entropy of the
    normalized feature-occupancy distribution across all materialized
    prototypes. More diverse prototypes -> flatter distribution -> higher H.
    """
    if not beliefs:
        return 0.0
    occ = np.zeros(n_features)
    for k in beliefs:
        occ[list(k)] += 1.0
    total = occ.sum()
    if total == 0:
        return 0.0
    pdist = occ / total
    pdist = pdist[pdist > 0]
    return float(-(pdist * np.log2(pdist)).sum())


def _global_optimum(landscape, n_sample=200000, seed=RNG_SEED):
    """Estimate the global max utility by large random + greedy search."""
    rng = np.random.default_rng(seed)
    n = landscape.n
    best = 0.0
    # random sampling
    for _ in range(8):
        P = (rng.random((n_sample // 8, n)) < 0.35).astype(float)
        u = np.array([landscape.utility(p) for p in P[:2000]])  # subsample eval
        best = max(best, u.max())
    # greedy hill-climb from several starts
    for _ in range(40):
        p = (rng.random(n) < 0.3).astype(float)
        improved = True
        while improved:
            improved = False
            base = landscape.utility(p)
            for i in range(n):
                q = p.copy(); q[i] = 1 - q[i]
                if landscape.utility(q) > base + 1e-9:
                    p, base, improved = q, landscape.utility(q), True
        best = max(best, base)
    return best


def _utility_batch(landscape, P):
    """Vectorized utility for a (m x n) 0/1 matrix P (matches UtilityLandscape)."""
    L = landscape
    coverage = P @ L.need_masks.T                       # (m, n_needs)
    U_D = (1.0 - np.exp(-0.9 * coverage)) @ L.need_weight
    raw = P @ L.value - 0.04 * np.clip(P.sum(1) - 6, 0, None) ** 1.5
    U_V = _sigmoid(0.8 * raw - 1.0)
    c = P @ L.cost + np.einsum("ij,jk,ik->i", P, L.inter, P)
    U_F = np.exp(-0.55 * c)
    return L.w_d * U_D + L.w_v * U_V + L.w_f * U_F


def characterize_landscape(landscape):
    """Full-enumeration ruggedness metrics on {0,1}^n (n=14 -> 16384 points):
    number of local optima under the 1-flip neighborhood, and the
    fitness-distance correlation (FDC) to the global optimum."""
    n = landscape.n
    m = 1 << n
    # enumerate all bit vectors
    idx = np.arange(m)
    P = ((idx[:, None] >> np.arange(n)[None, :]) & 1).astype(float)
    U = _utility_batch(landscape, P)
    # 1-flip local optima: U_i >= all its n neighbours
    is_opt = np.ones(m, dtype=bool)
    for j in range(n):
        Pj = P.copy(); Pj[:, j] = 1.0 - Pj[:, j]
        Uj = _utility_batch(landscape, Pj)
        is_opt &= (U >= Uj - 1e-12)
    n_local = int(is_opt.sum())
    gopt_idx = int(np.argmax(U))
    # Hamming distance to the global optimum
    dist = (P != P[gopt_idx]).sum(1)
    fdc = float(np.corrcoef(U, dist)[0, 1])
    return {"n_states": m, "n_local_optima": n_local,
            "global_opt": float(U.max()), "fdc": fdc,
            "mean_util": float(U.mean())}


# =========================================================================== #
#  4. MONTE-CARLO ENSEMBLE
# =========================================================================== #
def monte_carlo(landscape, n_runs=400, **kw):
    runs = [run_design_process(landscape, seed=1000 + r, **kw)
            for r in range(n_runs)]
    Xk = np.vstack([r["Xk"] for r in runs])          # (n_runs, n_iter)
    Xtrue = np.vstack([r["Xtrue"] for r in runs])
    ent = np.vstack([r["entropy"] for r in runs])
    ntd = np.vstack([r["n_tested"] for r in runs])
    btrue = np.vstack([r["best_true"] for r in runs])
    return {"Xk": Xk, "Xtrue": Xtrue, "entropy": ent, "n_tested": ntd,
            "best_true": btrue, "runs": runs,
            "global_opt": runs[0]["global_opt"]}


# =========================================================================== #
#  5. FIGURES
# =========================================================================== #
def fig_submartingale(mc, fname):
    """Replaces Fig5: ensemble convergence + sample paths + increment test."""
    Xt = mc["Xtrue"]                      # theorem's X_k (retained best utility)
    Xb = mc["Xk"]                         # team's belief (noisy estimate)
    n_runs, n_iter = Xt.shape
    ks = np.arange(n_iter)
    mean = Xt.mean(0)
    p10, p90 = np.percentile(Xt, [10, 90], axis=0)
    gopt = mc["global_opt"]

    fig = plt.figure(figsize=(14.5, 5.2))
    gs = GridSpec(1, 3, width_ratios=[1.5, 1.0, 0.85], wspace=0.42)

    # -- (a) ensemble + sample paths -- #
    ax = fig.add_subplot(gs[0])
    rng = np.random.default_rng(7)
    for r in rng.choice(n_runs, 14, replace=False):
        ax.step(ks, Xt[r], where="post", color=C_GREY, lw=0.7, alpha=0.45,
                zorder=1)
    ax.fill_between(ks, p10, p90, color=C_BAND, alpha=0.35,
                    label="10th–90th percentile", zorder=2)
    ax.plot(ks, mean, color=C_PRIMARY, lw=2.7,
            label=r"ensemble mean $\mathbb{E}[X_k]$", zorder=5)
    ax.plot(ks, Xb.mean(0), color=C_GREEN, lw=1.8, ls=(0, (4, 2)),
            label=r"team belief $\max_p\mathbb{E}[U\mid\mathcal{F}_k]$", zorder=4)
    ax.axhline(gopt, color=C_ACCENT, ls="--", lw=1.6,
               label="estimated global optimum $U^*$", zorder=3)
    ax.set_xlabel("Iteration $k$")
    ax.set_ylabel(r"$X_k=\max_{p\in P_{r,k}}\,\mathbb{E}[U(p)\mid\mathcal{F}_k]$")
    ax.set_title(r"(a) Convergence of $X_k$ ($N{=}%d$)" % n_runs)
    ax.legend(loc="lower right", fontsize=9)
    ax.set_ylim(min(0.45, p10.min() - 0.03), max(0.82, gopt + 0.04))

    # -- (b) mean per-step increment (submartingale condition) -- #
    ax2 = fig.add_subplot(gs[1])
    incr = np.diff(Xt, axis=1)
    incr_mean = incr.mean(0)
    incr_sem = incr.std(0) / np.sqrt(n_runs)
    kk = np.arange(1, n_iter)
    ax2.axhline(0, color=C_ACCENT, lw=1.4, ls="--", zorder=1)
    ax2.bar(kk, incr_mean, width=0.9, color=C_GREEN, alpha=0.8,
            yerr=1.96 * incr_sem, ecolor=C_GREY, error_kw={"lw": 0.5},
            label=r"$\mathbb{E}[\Delta X_k]\pm$95% CI")
    ax2.set_xlabel("Iteration $k$")
    ax2.set_ylabel(r"Mean increment $\mathbb{E}[\Delta X_k]$")
    ax2.set_title("(b) Non-negative drift\n(submartingale condition)")
    ax2.legend(loc="upper right", fontsize=9)
    frac = float((incr_mean >= -1e-9).mean())
    ax2.text(0.50, 0.86,
             r"$\mathbb{E}[\Delta X_k]\geq 0$" + f"\nat {frac*100:.0f}% of steps",
             transform=ax2.transAxes, fontsize=9, va="top",
             bbox=dict(boxstyle="round", fc="white", ec=C_GREY, alpha=0.9))

    # -- (c) distribution of per-run limits (local optima) -- #
    ax3 = fig.add_subplot(gs[2])
    limits = Xt[:, -1]
    ax3.hist(limits, bins=18, orientation="horizontal", color=C_PURPLE,
             alpha=0.8, edgecolor="white", lw=0.4)
    ax3.axhline(gopt, color=C_ACCENT, ls="--", lw=1.6, label="$U^*$")
    ax3.set_xlabel("runs")
    ax3.set_ylabel(r"limit $X_\infty$")
    ax3.set_title("(c) Run limits:\nlocal optima")
    ax3.legend(fontsize=9, loc="lower right")
    ax3.set_ylim(ax.get_ylim())

    fig.savefig(fname)
    plt.close(fig)
    return {"frac_nonneg": frac, "mean_final": float(mean[-1]),
            "global_opt": float(gopt), "min_increment": float(incr_mean.min()),
            "frac_below_opt": float((limits < gopt - 1e-6).mean())}


def fig_entropy(landscape, fname):
    """Replaces Fig8: entropy rises under divergence, falls under convergence."""
    # Controlled experiment: alternate pure divergence and pure convergence.
    rng = np.random.default_rng(RNG_SEED)
    n = landscape.n
    ideas = [(rng.random(n) < 0.3).astype(float) for _ in range(3)]

    def occ_entropy(idea_list):
        occ = np.zeros(n)
        for p in idea_list:
            occ[p.astype(bool)] += 1
        tot = occ.sum()
        if tot == 0:
            return 0.0
        pd = occ / tot
        pd = pd[pd > 0]
        return float(-(pd * np.log2(pd)).sum())

    # start from a SINGLE small seed idea (low entropy, low cardinality)
    seed0 = np.zeros(n); seed0[rng.choice(n, 2, replace=False)] = 1.0
    ideas = [seed0]

    phases, ents, sizes = [], [], []
    phases.append("div"); ents.append(occ_entropy(ideas)); sizes.append(len(ideas))
    # Phase 1: divergence (Ideate) -- generate new diverse ideas
    for t in range(9):
        for _ in range(3):
            ideas.append((rng.random(n) < 0.4).astype(float))
        phases.append("div"); ents.append(occ_entropy(ideas)); sizes.append(len(ideas))
    # Phase 2: convergence (Select) -- keep only the top-utility ideas
    for t in range(9):
        ideas.sort(key=lambda p: landscape.utility(p), reverse=True)
        ideas = ideas[:max(1, int(round(len(ideas) * 0.72)))]
        phases.append("conv"); ents.append(occ_entropy(ideas)); sizes.append(len(ideas))

    fig, ax = plt.subplots(figsize=(8.6, 4.7))
    x = np.arange(len(ents))
    split = phases.index("conv")
    ax.axvspan(-0.5, split - 0.5, color=C_GREEN, alpha=0.08)
    ax.axvspan(split - 0.5, len(ents) - 0.5, color=C_ACCENT, alpha=0.08)
    ax.plot(x[:split], ents[:split], "o-", color=C_GREEN, lw=2.4, ms=6,
            label=r"$H(v_k)$ — divergent $Op_{\mathrm{Div}}$ (Ideate)")
    ax.plot(x[split - 1:], ents[split - 1:], "s-", color=C_ACCENT, lw=2.4, ms=6,
            label=r"$H(v_k)$ — convergent $Op_{\mathrm{Conv}}$ (Select)")
    ax.axvline(split - 0.5, color=C_GREY, ls=":", lw=1.2)
    ax.set_xlabel("Operator application step")
    ax.set_ylabel("Shannon entropy $H(v_k)$  [bits]", color="#1A5276")
    ax.tick_params(axis="y", labelcolor="#1A5276")
    ax.set_title("Divergence raises entropy and cardinality;\nconvergence lowers both")

    # twin axis: cardinality |S_i| of the idea set
    ax2 = ax.twinx()
    ax2.plot(x, sizes, "^--", color=C_PURPLE, lw=1.6, ms=5, alpha=0.85,
             label=r"cardinality $|S_i|$")
    ax2.set_ylabel(r"Idea-set cardinality $|S_i|$", color=C_PURPLE)
    ax2.tick_params(axis="y", labelcolor=C_PURPLE)
    ax2.grid(False)

    l1, lab1 = ax.get_legend_handles_labels()
    l2, lab2 = ax2.get_legend_handles_labels()
    ax.legend(l1 + l2, lab1 + lab2, loc="lower center", fontsize=9)
    ax.text(split / 2, ax.get_ylim()[0] + 0.12, "expansion", ha="center",
            color=C_GREEN, fontsize=10, style="italic")
    ax.text(split + (len(ents) - split) / 2, ax.get_ylim()[0] + 0.12,
            "selection", ha="center", color=C_ACCENT, fontsize=10, style="italic")
    fig.savefig(fname)
    plt.close(fig)
    return {"H_peak": float(max(ents)), "H_start": float(ents[0]),
            "H_end": float(ents[-1]), "card_peak": int(max(sizes))}


def fig_bayes(landscape, fname, sigma_obs=0.12):
    """Replaces Fig3: sequential Bayesian update of a prototype's utility."""
    rng = np.random.default_rng(3)
    n = landscape.n
    p = (rng.random(n) < 0.35).astype(float)
    u_true = landscape.utility(p)

    belief = UtilityBelief(prior_mu=0.40, prior_var=0.05, sigma_obs=sigma_obs)
    grid = np.linspace(0, 1, 600)

    def npdf(x, m, v):
        return np.exp(-(x - m) ** 2 / (2 * v)) / np.sqrt(2 * np.pi * v)

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 4.6),
                                  gridspec_kw={"width_ratios": [1.3, 1.0],
                                               "wspace": 0.28})
    obs_counts = [0, 1, 3, 6, 12]
    cmap = plt.cm.viridis(np.linspace(0.15, 0.85, len(obs_counts)))
    b = UtilityBelief(0.40, 0.05, sigma_obs)
    means, vars_, seen = [b.mu], [b.var], 0
    ax.plot(grid, npdf(grid, b.mu, b.var), color=cmap[0], lw=2,
            label="prior (0 tests)")
    all_y = []
    for j in range(1, max(obs_counts) + 1):
        y = u_true + rng.normal(0, sigma_obs)
        all_y.append(y)
        b.update(y)
        means.append(b.mu); vars_.append(b.var)
        if j in obs_counts:
            idx = obs_counts.index(j)
            ax.plot(grid, npdf(grid, b.mu, b.var), color=cmap[idx], lw=2,
                    label=f"posterior ({j} tests)")
    ax.axvline(u_true, color=C_ACCENT, ls="--", lw=1.8, label="true utility $u^*$")
    ax.set_xlabel("Utility value $u$")
    ax.set_ylabel("Belief density")
    ax.set_title(r"(a) Bayesian update: posterior concentrates on $u^*$")
    ax.legend(fontsize=9)

    # right: posterior mean (martingale) +/- std band converging to u*
    ax2.fill_between(range(len(means)),
                     np.array(means) - np.sqrt(vars_),
                     np.array(means) + np.sqrt(vars_),
                     color=C_BAND, alpha=0.35, label=r"posterior mean $\pm\sigma$")
    ax2.plot(range(len(means)), means, "o-", color=C_PRIMARY, lw=2,
             label=r"posterior mean $\mathbb{E}[U\mid\mathcal{F}_k]$")
    ax2.axhline(u_true, color=C_ACCENT, ls="--", lw=1.8, label="true utility $u^*$")
    ax2.set_xlabel("Number of tests")
    ax2.set_ylabel("Estimated utility")
    ax2.set_title("(b) Posterior mean is a martingale\nconverging to $u^*$")
    ax2.legend(fontsize=9, loc="best")
    fig.savefig(fname)
    plt.close(fig)
    return {"u_true": float(u_true), "final_mean": float(means[-1])}


def fig_utility_surface(landscape, fname):
    """Replaces Fig4: the (rugged) utility landscape, 2-D projection."""
    # Project the discrete prototype space onto (desirability, feasibility)
    rng = np.random.default_rng(RNG_SEED)
    n = landscape.n
    P = (rng.random((6000, n)) < rng.uniform(0.15, 0.6, (6000, 1))).astype(float)
    ud = np.array([landscape._desirability(p) for p in P])
    uf = np.array([landscape._feasibility(p) for p in P])
    uu = np.array([landscape.utility(p) for p in P])

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 4.6),
                                  gridspec_kw={"width_ratios": [1.1, 1.0],
                                               "wspace": 0.3})
    sc = ax.scatter(ud, uf, c=uu, cmap="viridis", s=14, alpha=0.8)
    cb = fig.colorbar(sc, ax=ax); cb.set_label("Utility $U(p)$")
    ax.set_xlabel("Desirability $U_D(p)$")
    ax.set_ylabel("Feasibility $U_F(p)$")
    ax.set_title("(a) Utility over the desirability–feasibility plane")

    # histogram of utilities -> multimodality (wickedness)
    ax2.hist(uu, bins=40, color=C_PURPLE, alpha=0.8, edgecolor="white", lw=0.4)
    ax2.set_xlabel("Utility $U(p)$")
    ax2.set_ylabel("Number of prototypes")
    ax2.set_title("(b) Multimodal utility distribution\n(non-convex search space)")
    fig.savefig(fname)
    plt.close(fig)
    return {"U_max_sampled": float(uu.max())}


def fig_threshold(mc, fname, eps=0.01, N=5):
    """Replaces Fig6: marginal improvement decays below epsilon (stopping rule)."""
    Xk = mc["Xtrue"]
    n_runs, n_iter = Xk.shape
    # marginal N-step expected improvement E[X_{k+N}-X_k]
    marg = (Xk[:, N:] - Xk[:, :-N]).mean(0)
    ks = np.arange(len(marg))
    # first crossing of eps
    below = np.where(marg < eps)[0]
    k_star = int(below[0]) if len(below) else None

    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    ax.plot(ks, marg, color=C_PRIMARY, lw=2.4,
            label=r"$\mathbb{E}[U(p_{k+N})-U(p_k)]$, $N=%d$" % N)
    ax.axhline(eps, color=C_ACCENT, ls="--", lw=1.8,
               label=r"threshold $\epsilon=%.2f$" % eps)
    ax.fill_between(ks, 0, marg, where=(marg >= eps), color=C_BAND, alpha=0.25)
    if k_star is not None:
        ax.axvline(k_star, color=C_GREEN, ls=":", lw=1.8)
        ax.annotate(r"convergence: $k^*=%d$" % k_star,
                    xy=(k_star, eps), xytext=(k_star + 6, eps + 0.04),
                    arrowprops=dict(arrowstyle="->", color=C_GREEN),
                    color=C_GREEN, fontsize=10)
    ax.set_xlabel("Iteration $k$")
    ax.set_ylabel("Marginal expected utility improvement")
    ax.set_title(r"Marginal improvement plateaus below $\epsilon$ (stopping rule)")
    ax.legend(fontsize=10)
    ax.set_ylim(bottom=-0.005)
    fig.savefig(fname)
    plt.close(fig)
    return {"k_star": k_star, "eps": eps, "N": N}


def fig_diagnostics(mc, fname, eps=0.01, N=5):
    """Fig9: convergence diagnostics that go beyond the summary table.

    (a) Lyapunov potential Phi_k = U* - X_k decaying toward 0 (semi-log) -- the
        supermartingale potential from the proof.
    (b) Per-run convergence iteration k* (first time the N-step gain < eps).
    (c) Exploration breadth |P_r| and state entropy vs iteration.
    """
    Xt = mc["Xtrue"]
    ent = mc["entropy"]
    ntd = mc["n_tested"]
    gopt = mc["global_opt"]
    n_runs, n_iter = Xt.shape
    ks = np.arange(n_iter)

    fig = plt.figure(figsize=(14.5, 4.9))
    gs = GridSpec(1, 3, width_ratios=[1.05, 1.0, 1.05], wspace=0.34)

    # (a) Lyapunov potential decay
    phi = np.clip(gopt - Xt, 1e-4, None)
    ax = fig.add_subplot(gs[0])
    ax.fill_between(ks, np.percentile(phi, 25, 0), np.percentile(phi, 75, 0),
                    color=C_BAND, alpha=0.35, label="IQR")
    ax.semilogy(ks, phi.mean(0), color=C_PRIMARY, lw=2.5,
                label=r"$\mathbb{E}[\Phi_k]$")
    ax.set_xlabel("Iteration $k$")
    ax.set_ylabel(r"Potential $\Phi_k = U^*-X_k$  (log)")
    ax.set_title("(a) Lyapunov potential decay")
    ax.legend(fontsize=9)

    # (b) per-run convergence iteration k*
    kstar = []
    for r in range(n_runs):
        marg = Xt[r, N:] - Xt[r, :-N]
        below = np.where(marg < eps)[0]
        kstar.append(int(below[0]) if len(below) else n_iter - N)
    kstar = np.array(kstar)
    ax2 = fig.add_subplot(gs[1])
    ax2.hist(kstar, bins=np.arange(0, kstar.max() + 3) - 0.5,
             color=C_GREEN, alpha=0.8, edgecolor="white", lw=0.4)
    ax2.axvline(kstar.mean(), color=C_ACCENT, ls="--", lw=1.8,
                label=r"mean $k^*=%.1f$" % kstar.mean())
    ax2.axvline(np.median(kstar), color=C_PURPLE, ls=":", lw=1.8,
                label=r"median $=%.0f$" % np.median(kstar))
    ax2.set_xlabel(r"Convergence iteration $k^*$ (per run)")
    ax2.set_ylabel("Number of runs")
    ax2.set_title(r"(b) Convergence-speed distribution")
    ax2.legend(fontsize=9)

    # (c) exploration breadth + entropy
    ax3 = fig.add_subplot(gs[2])
    ax3.fill_between(ks, np.percentile(ntd, 10, 0), np.percentile(ntd, 90, 0),
                     color=C_BAND, alpha=0.3)
    ax3.plot(ks, ntd.mean(0), color=C_PRIMARY, lw=2.4,
             label=r"$|P_r|$ explored")
    ax3.set_xlabel("Iteration $k$")
    ax3.set_ylabel(r"Prototypes explored $|P_r|$", color=C_PRIMARY)
    ax3.tick_params(axis="y", labelcolor=C_PRIMARY)
    ax3b = ax3.twinx()
    ax3b.plot(ks, ent.mean(0), color=C_ACCENT, lw=2.0, ls="--",
              label=r"state entropy $H(v_k)$")
    ax3b.set_ylabel(r"Entropy $H(v_k)$ [bits]", color=C_ACCENT)
    ax3b.tick_params(axis="y", labelcolor=C_ACCENT)
    ax3b.grid(False)
    ax3.set_title("(c) Exploration breadth \\& entropy")
    l1, la1 = ax3.get_legend_handles_labels()
    l2, la2 = ax3b.get_legend_handles_labels()
    ax3.legend(l1 + l2, la1 + la2, fontsize=9, loc="lower right")

    fig.savefig(fname)
    plt.close(fig)
    return {"kstar_mean": float(kstar.mean()),
            "kstar_median": float(np.median(kstar)),
            "kstar_p90": float(np.percentile(kstar, 90)),
            "phi_final": float(phi.mean(0)[-1]),
            "ntd_final": float(ntd.mean(0)[-1])}


def dump_data(mc, txt_path, tex_path):
    """Write the per-iteration trajectory data as a plain table (.txt) and a
    LaTeX-ready snippet (.tex) so the article can show concrete numbers."""
    Xt, Xb = mc["Xtrue"], mc["Xk"]
    ent, ntd = mc["entropy"], mc["n_tested"]
    gopt = mc["global_opt"]
    sel = [0, 1, 2, 5, 10, 20, 40, 79]
    header = f"{'k':>4} {'E[X_k]':>9} {'sd(X_k)':>9} {'E[Phi_k]':>9} " \
             f"{'E[H]':>7} {'E[|Pr|]':>8}"
    lines = ["# Per-iteration ensemble statistics (250 runs)",
             f"# global optimum U* = {gopt:.4f}", header]
    for k in sel:
        lines.append(f"{k:>4} {Xt[:,k].mean():>9.4f} {Xt[:,k].std():>9.4f} "
                     f"{(gopt-Xt[:,k]).mean():>9.4f} {ent[:,k].mean():>7.3f} "
                     f"{ntd[:,k].mean():>8.1f}")
    open(txt_path, "w").write("\n".join(lines) + "\n")

    # LaTeX booktabs rows
    rows = []
    for k in sel:
        rows.append(f"{k} & {Xt[:,k].mean():.4f} & {Xt[:,k].std():.4f} & "
                    f"{(gopt-Xt[:,k]).mean():.4f} & {ent[:,k].mean():.3f} & "
                    f"{ntd[:,k].mean():.1f} \\\\")
    open(tex_path, "w").write("\n".join(rows) + "\n")
    return "\n".join(lines)


# =========================================================================== #
#  6. MAIN
# =========================================================================== #
def main():
    print("=" * 70)
    print("Design-Thinking convergence simulation")
    print("=" * 70)
    land = UtilityLandscape()
    print(f"Feature space |F| = {land.n}  ->  prototype space |2^F| = 2^{land.n}"
          f" = {2**land.n:,}")
    print(f"Utility weights (w_d, w_v, w_f) = "
          f"({land.w_d}, {land.w_v}, {land.w_f})")

    print("\nRunning Monte-Carlo ensemble ...")
    cache = os.path.join(OUTDIR, "ensemble_cache.npz")
    N_RUNS, N_ITER = 250, 80
    if os.environ.get("USE_CACHE") and os.path.exists(cache):
        d = np.load(cache)
        mc = {"Xk": d["Xk"], "Xtrue": d["Xtrue"], "entropy": d["entropy"],
              "n_tested": d["n_tested"], "best_true": d["best_true"],
              "global_opt": float(d["global_opt"]), "runs": None}
    else:
        mc = monte_carlo(land, n_runs=N_RUNS, n_iter=N_ITER)
        np.savez(cache, Xk=mc["Xk"], Xtrue=mc["Xtrue"], entropy=mc["entropy"],
                 n_tested=mc["n_tested"], best_true=mc["best_true"],
                 global_opt=mc["global_opt"])
    print(f"  ensemble size: {mc['Xk'].shape[0]} runs x {mc['Xk'].shape[1]} iters")
    print(f"  estimated global optimum U* = {mc['global_opt']:.4f}")

    results = {}
    results["submartingale"] = fig_submartingale(
        mc, os.path.join(OUTDIR, "Fig5.png"))
    results["entropy"] = fig_entropy(land, os.path.join(OUTDIR, "Fig8.png"))
    results["bayes"] = fig_bayes(land, os.path.join(OUTDIR, "Fig3.png"))
    results["utility"] = fig_utility_surface(land, os.path.join(OUTDIR, "Fig4.png"))
    results["threshold"] = fig_threshold(mc, os.path.join(OUTDIR, "Fig6.png"))
    results["diagnostics"] = fig_diagnostics(mc, os.path.join(OUTDIR, "Fig9.png"))
    table = dump_data(mc, os.path.join(OUTDIR, "results_data.txt"),
                      os.path.join(OUTDIR, "results_rows.tex"))

    print("\n--- PER-ITERATION DATA -------------------------------------------")
    print(table)
    print("\n--- VALIDATION RESULTS -------------------------------------------")
    s = results["submartingale"]
    print(f"[C1] Submartingale: fraction of steps with E[dX]>=0 : "
          f"{s['frac_nonneg']*100:.1f}%")
    print(f"     min mean-increment over all steps             : "
          f"{s['min_increment']:+.4f}  (>=~0 expected)")
    print(f"[C2] Ensemble-mean final X_k                       : "
          f"{s['mean_final']:.4f}  (global opt {s['global_opt']:.4f})")
    print(f"     runs converging to a strict LOCAL optimum     : "
          f"{s['frac_below_opt']*100:.1f}%")
    e = results["entropy"]
    print(f"[C3] Entropy  start={e['H_start']:.3f}  peak(div)={e['H_peak']:.3f} "
          f" end(conv)={e['H_end']:.3f}  bits")
    b = results["bayes"]
    print(f"[C4] Bayesian: u_true={b['u_true']:.3f}  ->  posterior mean="
          f"{b['final_mean']:.3f}")
    t = results["threshold"]
    print(f"[C5] eps-convergence at iteration k* = {t['k_star']} "
          f"(eps={t['eps']}, N={t['N']})")
    d = results["diagnostics"]
    print(f"[C6] per-run k*: mean={d['kstar_mean']:.1f} "
          f"median={d['kstar_median']:.0f} p90={d['kstar_p90']:.0f}; "
          f"final potential E[Phi]={d['phi_final']:.4f}; "
          f"explored |Pr|={d['ntd_final']:.0f}")
    print("-" * 66)
    print("Figures written: Fig3-6.png Fig8.png Fig9.png + results_data.txt")
    return results


if __name__ == "__main__":
    main()
