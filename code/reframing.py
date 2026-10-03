#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Does the reframing criterion of Eq. (6) actually pay?  A policy comparison.

The convergence theorem is silent about WHEN to abandon a frame.  Proposition 3
gives a one-step-lookahead index rule,

        EI_k - c_iter  <  G_ref - c_ref                                    (6)

and the reviewer's fair question is whether a team following it does better than
one that over-exploits (never reframes) or over-explores (reframes on a fixed
schedule).  This script answers that question *in silico*.

Setup ("bandit over frames", Section 5.6).
------------------------------------------
A run has a fixed total budget of T rounds.  The world contains M frames; each
frame is an independent UtilityLandscape (its own feature semantics and utility,
hence its own ceiling U*_f).  Within a frame the team runs the bounded
Ideate-Prototype-Test loop of Section 4.  A reframe costs c_ref_rounds rounds of
the budget (re-empathising, rebuilding context) and moves the team to a fresh,
previously unvisited frame, with beliefs reset.  The run's payoff is the best
TRUE utility ever attained by an incumbent in any frame -- the team may ship the
best thing it found.

Policies compared
-----------------
  eq6      : reframe exactly when inequality (6) holds.  EI_k is the posterior
             expected improvement of testing one more candidate; G_ref is the
             *frame-level* expected improvement under a hierarchical prior over
             frame ceilings, updated from the frames visited so far.  (Note the
             pleasing symmetry: G_ref is an EI one level up.)
  eq6ramp  : the same rule, made less myopic: the cost of switching is charged
             not only the one-off c_ref but also the expected RAMP-UP -- the
             rounds a team needs in a fresh frame before it reaches the level it
             already holds -- estimated from the team's own history.  This is the
             two-level (meta-bandit) reading of the criterion.
  exploit  : never reframe (perfect the first frame).
  explore  : reframe every kappa rounds regardless of evidence.
  matched  : reframe at random rounds, with the per-run NUMBER of reframes drawn
             to match the eq6 policy's realised distribution.  This is the
             important control: it isolates the value of reframing at the RIGHT
             TIME from the value of reframing the right NUMBER OF TIMES.

Dependencies: numpy, matplotlib (+ simulation.py in the same directory).
"""
import json
import os
from math import erf

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import simulation as S

OUTDIR = os.path.dirname(os.path.abspath(__file__))
OUT_JSON = os.path.join(OUTDIR, "reframing_results.json")

mpl.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
    "font.family": "serif", "font.size": 12.5,
    "axes.titlesize": 13.5, "axes.titleweight": "bold", "axes.labelsize": 12.5,
    "xtick.labelsize": 11.5, "ytick.labelsize": 11.5, "legend.fontsize": 11.5,
    "axes.edgecolor": "#444444", "axes.linewidth": 0.9,
    "axes.grid": True, "grid.color": "#D5D8DC", "grid.linewidth": 0.7,
    "legend.frameon": True, "legend.framealpha": 0.92,
    "mathtext.fontset": "cm",
})
C_EQ6, C_EXPLOIT, C_EXPLORE, C_MATCH = "#1E8449", "#C0392B", "#B9770E", "#2C5F8A"

# ---- experiment constants -------------------------------------------------- #
T_BUDGET = 150          # total rounds available to a run
M_FRAMES = 6            # frames in the world
C_REF_ROUNDS = 10       # a reframe costs this many rounds of budget
LAMBDA = 0.002          # marginal cost of one iteration, in utility units
KAPPA = 15              # 'explore' policy: reframe every kappa rounds
N_RUNS = 600
PRIOR_MU, PRIOR_VAR, SIGMA_OBS = 0.5, 0.08, 0.10
IDEAS, RETEST, R_MAX = 5, 3, 2
# cross-project hierarchical prior over a frame's ceiling
CEIL_M0, CEIL_TAU2, CEIL_OBS2 = 0.75, 0.05 ** 2, 0.03 ** 2

SQRT2PI = np.sqrt(2.0 * np.pi)


def _norm_pdf(z):
    return np.exp(-0.5 * z * z) / SQRT2PI


def _norm_cdf(z):
    if np.isscalar(z):
        return 0.5 * (1.0 + erf(z / np.sqrt(2.0)))
    return np.array([0.5 * (1.0 + erf(float(x) / np.sqrt(2.0))) for x in z])


def expected_improvement(mu, sd, incumbent):
    """Standard EI of a Gaussian belief N(mu, sd^2) over an incumbent value."""
    sd = max(float(sd), 1e-9)
    z = (mu - incumbent) / sd
    return float((mu - incumbent) * _norm_cdf(z) + sd * _norm_pdf(z))


class Frame:
    """One framing epoch: a landscape plus the team's state inside it."""

    def __init__(self, landscape, rng):
        self.L = landscape
        self.rng = rng
        self.n = landscape.n
        self.beliefs = {}
        self.true_util = {}
        self.proxy = {}
        self.best_true = 0.0
        for _ in range(IDEAS):
            self._add((rng.random(self.n) < 0.3).astype(float))

    def _key(self, p):
        return frozenset(np.flatnonzero(p).tolist())

    def _vec(self, k):
        v = np.zeros(self.n)
        if k:
            v[list(k)] = 1.0
        return v

    def _add(self, p):
        k = self._key(p)
        if k not in self.beliefs and k not in self.proxy:
            tu = self.L.utility(p)
            self.true_util[k] = tu
            self.proxy[k] = tu + self.rng.normal(0, 0.10)

    def step(self):
        """One Ideate -> Prototype -> Test round; returns the new retained best."""
        rng = self.rng
        if self.beliefs:
            frontier = sorted(self.beliefs, key=lambda kk: self.beliefs[kk].mu,
                              reverse=True)[:RETEST]
            base = self._vec(frontier[rng.integers(len(frontier))])
        else:
            base = (rng.random(self.n) < 0.3).astype(float)
        for _ in range(IDEAS):
            child = base.copy()
            flips = rng.integers(1, R_MAX + 1)
            idx = rng.choice(self.n, size=flips, replace=False)
            child[idx] = 1.0 - child[idx]
            self._add(child)

        if self.proxy:
            chosen = max(self.proxy, key=self.proxy.get)
            del self.proxy[chosen]
            self.beliefs[chosen] = S.UtilityBelief(PRIOR_MU, PRIOR_VAR, SIGMA_OBS)
            self.beliefs[chosen].update(self.true_util[chosen] +
                                        rng.normal(0, SIGMA_OBS))
        if self.beliefs:
            top = sorted(self.beliefs, key=lambda kk: self.beliefs[kk].mu,
                         reverse=True)[:RETEST]
            for kk in top:
                self.beliefs[kk].update(self.true_util[kk] +
                                        rng.normal(0, SIGMA_OBS))
            inc = max(self.beliefs, key=lambda kk: self.beliefs[kk].mu)
            self.best_true = max(self.best_true, self.true_util[inc])
        return self.best_true

    def incumbent_mu(self):
        return max((b.mu for b in self.beliefs.values()), default=PRIOR_MU)

    def EI(self):
        """
        Posterior expected improvement of testing one more candidate.

        EI is increasing in the mean at fixed variance, so it suffices to
        evaluate the best untested candidate (all share the prior variance) and
        the few most promising live prototypes; scanning the whole pool would
        change nothing but cost.
        """
        inc = self.incumbent_mu()
        best = 0.0
        if self.proxy:
            g = max(self.proxy.values())
            best = expected_improvement(min(max(g, 0.0), 1.0),
                                        np.sqrt(PRIOR_VAR), inc)
        if self.beliefs:
            top = sorted(self.beliefs.values(), key=lambda b: b.mu,
                         reverse=True)[:RETEST]
            for b in top:
                best = max(best, expected_improvement(b.mu, np.sqrt(b.var), inc))
        return best


class CeilingPrior:
    """
    Hierarchical Normal-Normal prior over an unvisited frame's attainable
    outcome.  `m0` and `tau2` are the team's CROSS-PROJECT prior; deliberately
    wrong values are what the G_ref-misspecification study sweeps.
    """

    def __init__(self, m0=None, tau2=None):
        self.mu = CEIL_M0 if m0 is None else float(m0)
        self.var = CEIL_TAU2 if tau2 is None else float(tau2)

    def update(self, observed_ceiling):
        p0, p = 1.0 / self.var, 1.0 / CEIL_OBS2
        self.var = 1.0 / (p0 + p)
        self.mu = self.var * (p0 * self.mu + p * observed_ceiling)

    def G_ref(self, retained_best):
        """Frame-level expected improvement: E[(ceiling - Y_k)^+]."""
        return expected_improvement(self.mu, np.sqrt(self.var + CEIL_TAU2),
                                    retained_best)


_POOL = {}


def _landscape(idx):
    """Cached pool of distinct frames (building one is not free)."""
    if idx not in _POOL:
        _POOL[idx] = S.UtilityLandscape(seed=104729 + 13 * idx)
    return _POOL[idx]


N_POOL = 40


def make_world(seed):
    """M independent frames (different feature semantics -> different ceilings)."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(N_POOL, size=M_FRAMES, replace=False)
    return [_landscape(int(i)) for i in idx]


def run_policy(policy, seed, n_reframes_target=None, record=False,
               prior_m0=None, prior_tau2=None, learn_prior=True):
    rng = np.random.default_rng(seed)
    world = make_world(seed)
    order = rng.permutation(M_FRAMES)
    fi = 0
    frame = Frame(world[order[fi]], rng)
    prior = CeilingPrior(prior_m0, prior_tau2)

    Y = 0.0                      # best true utility found anywhere
    traj = []
    reframe_at = []
    ramp_history = []            # rounds spent per completed frame
    t_frame = 0                  # rounds spent in the current frame
    budget = T_BUDGET
    t = 0

    # 'matched' policy: pre-draw the rounds at which it will reframe
    matched_rounds = set()
    if policy == "matched" and n_reframes_target:
        cand = rng.choice(np.arange(2, T_BUDGET - 2), size=min(
            n_reframes_target, T_BUDGET - 5), replace=False)
        matched_rounds = set(int(c) for c in cand)

    while t < budget:
        Y = max(Y, frame.step())
        traj.append(Y)
        t += 1
        t_frame += 1

        do_reframe = False
        if fi + 1 < M_FRAMES and t < budget - C_REF_ROUNDS - 1:
            if policy == "eq6":
                ei = frame.EI()
                g = prior.G_ref(Y)
                do_reframe = (ei - LAMBDA) < (g - C_REF_ROUNDS * LAMBDA)
            elif policy == "eq6ramp":
                ei = frame.EI()
                g = prior.G_ref(Y)
                # expected ramp-up in a fresh frame, self-estimated
                ramp = (float(np.mean(ramp_history)) if ramp_history
                        else float(t_frame))
                do_reframe = (ei - LAMBDA) < (g - (C_REF_ROUNDS + ramp) * LAMBDA)
            elif policy == "explore":
                do_reframe = (t % KAPPA == 0)
            elif policy == "matched":
                do_reframe = (t in matched_rounds)
            elif policy == "exploit":
                do_reframe = False

        if do_reframe:
            if learn_prior:
                prior.update(frame.best_true)      # learn about frame ceilings
            ramp_history.append(t_frame)
            t_frame = 0
            reframe_at.append(t)
            for _ in range(C_REF_ROUNDS):          # the reframe consumes budget
                if t >= budget:
                    break
                traj.append(Y)
                t += 1
            fi += 1
            frame = Frame(world[order[fi]], rng)

    traj = np.array(traj[:T_BUDGET])
    if traj.size < T_BUDGET:
        traj = np.concatenate([traj, np.full(T_BUDGET - traj.size, traj[-1])])
    out = {"final": float(Y), "n_reframes": len(reframe_at)}
    if record:
        out["traj"] = traj
        out["reframe_at"] = reframe_at
    return out


def sweep_c_ref(c_ref_values, n_runs=120):
    """How does the advantage of Eq. (6) depend on the cost of reframing?"""
    global C_REF_ROUNDS
    saved = C_REF_ROUNDS
    rows = []
    for c in c_ref_values:
        C_REF_ROUNDS = c
        res = {}
        for pol in ("eq6", "eq6ramp", "exploit", "explore"):
            v = [run_policy(pol, 5000 + r)["final"] for r in range(n_runs)]
            res[pol] = (float(np.mean(v)),
                        float(1.96 * np.std(v, ddof=1) / np.sqrt(len(v))))
        rows.append({"c_ref": c, **res})
        print(f"  c_ref={c:2d}: eq6={res['eq6'][0]:.4f}  "
              f"eq6ramp={res['eq6ramp'][0]:.4f}  "
              f"exploit={res['exploit'][0]:.4f}  explore={res['explore'][0]:.4f}")
    C_REF_ROUNDS = saved
    return rows


def main():
    print("Policy comparison over %d runs, budget T=%d, M=%d frames, "
          "c_ref=%d rounds\n" % (N_RUNS, T_BUDGET, M_FRAMES, C_REF_ROUNDS))

    results, trajs = {}, {}
    # eq6 first, to learn its realised reframe count for the matched control
    r_eq6 = [run_policy("eq6", r, record=(r < 60)) for r in range(N_RUNS)]
    n_ref_eq6 = [d["n_reframes"] for d in r_eq6]
    results["eq6"] = [d["final"] for d in r_eq6]
    trajs["eq6"] = np.array([d["traj"] for d in r_eq6 if "traj" in d])

    for pol in ("eq6ramp", "exploit", "explore", "matched"):
        rs = []
        for r in range(N_RUNS):
            tgt = int(n_ref_eq6[r]) if pol == "matched" else None
            rs.append(run_policy(pol, r, n_reframes_target=tgt,
                                 record=(r < 60)))
        results[pol] = [d["final"] for d in rs]
        trajs[pol] = np.array([d["traj"] for d in rs if "traj" in d])

    summary = {}
    for pol, v in results.items():
        v = np.array(v)
        summary[pol] = {"mean": float(v.mean()),
                        "ci95": float(1.96 * v.std(ddof=1) / np.sqrt(v.size)),
                        "median": float(np.median(v)),
                        "p5": float(np.percentile(v, 5)),
                        "p95": float(np.percentile(v, 95))}
        print(f"  {pol:8s}  mean {summary[pol]['mean']:.4f} "
              f"+/- {summary[pol]['ci95']:.4f}   median "
              f"{summary[pol]['median']:.4f}")
    summary["eq6"]["mean_n_reframes"] = float(np.mean(n_ref_eq6))
    for pol in ("eq6ramp", "explore", "matched"):
        pass
    print(f"\n  eq6 reframes per run: mean {np.mean(n_ref_eq6):.2f}, "
          f"median {np.median(n_ref_eq6):.0f}")

    # paired differences vs eq6 (same world seeds -> paired comparison valid)
    for ref in ("eq6", "eq6ramp"):
        for pol in ("exploit", "explore", "matched"):
            d = np.array(results[ref]) - np.array(results[pol])
            t = d.mean() / (d.std(ddof=1) / np.sqrt(d.size))
            diff = np.abs(d) > 1e-12
            wins = float(np.mean(d[diff] > 0)) if diff.any() else float("nan")
            summary[f"{ref}_minus_{pol}"] = {
                "mean": float(d.mean()),
                "ci95": float(1.96 * d.std(ddof=1) / np.sqrt(d.size)),
                "t": float(t),
                "tie_rate": float(1 - diff.mean()),
                "win_rate_among_differing": wins}
            print(f"  {ref:7s} - {pol:8s} = {d.mean():+.4f} "
                  f"+/- {1.96*d.std(ddof=1)/np.sqrt(d.size):.4f}  "
                  f"(t={t:5.1f}; ties {100*(1-diff.mean()):.0f}%, "
                  f"wins {100*wins:.0f}% of the rest)")

    print("\n  cost-of-reframing sweep:")
    summary["c_ref_sweep"] = sweep_c_ref([4, 10, 20, 35])

    np.savez_compressed(os.path.join(OUTDIR, "reframing_trajs.npz"), **trajs)
    make_figure(results, trajs, summary,
                os.path.join(OUTDIR, "Fig13.png"))
    with open(OUT_JSON, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nwritten -> {OUT_JSON} and Fig13.png")


def make_figure(results, trajs, summary, fname):
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.0))
    order = ["exploit", "explore", "matched", "eq6", "eq6ramp"]
    labels = {"eq6": "Eq. (6) (myopic)", "eq6ramp": "Eq. (6) + ramp cost",
              "exploit": "never reframe (over-exploit)",
              "explore": "fixed schedule (over-explore)",
              "matched": "random, count-matched"}
    short = {"eq6": "Eq. (6)", "eq6ramp": "Eq. (6)\n+ ramp",
             "exploit": "never\nreframe", "explore": "fixed\nschedule",
             "matched": "random\n(matched)"}
    colors = {"eq6": C_EQ6, "eq6ramp": "#117A65", "exploit": C_EXPLOIT,
              "explore": C_EXPLORE, "matched": C_MATCH}

    # (a) final utility by policy
    ax = axes[0]
    xs = np.arange(len(order))
    means = [summary[p]["mean"] for p in order]
    cis = [summary[p]["ci95"] for p in order]
    ax.bar(xs, means, yerr=cis, capsize=5, width=0.62,
           color=[colors[p] for p in order], edgecolor="#333333", linewidth=0.8)
    ax.set_xticks(xs)
    ax.set_xticklabels([short[p] for p in order], fontsize=11)
    lo = min(means) - 4 * max(cis)
    ax.set_ylim(lo, max(means) + 5 * max(cis))
    ax.set_ylabel(r"best true utility attained,  $\mathbb{E}[Y_T]$")
    ax.set_title("(a) Outcome by reframing policy")
    for x, m, c in zip(xs, means, cis):
        ax.text(x, m + c + 0.0012, f"{m:.3f}", ha="center", fontsize=10.5)

    # (b) trajectories
    ax = axes[1]
    for p in order:
        Tj = trajs[p]
        if Tj.size == 0:
            continue
        m = Tj.mean(axis=0)
        se = 1.96 * Tj.std(axis=0, ddof=1) / np.sqrt(Tj.shape[0])
        ax.plot(m, color=colors[p], lw=2.0,
                label=labels[p])
        ax.fill_between(np.arange(m.size), m - se, m + se,
                        color=colors[p], alpha=0.16, linewidth=0)
    ax.set_xlabel("round (budget consumed, reframes included)")
    ax.set_ylabel(r"best true utility so far,  $Y_t$")
    ax.set_title("(b) Trajectories under a fixed budget")
    ax.legend(loc="lower right", fontsize=10.5)

    # (c) sensitivity to the cost of reframing
    ax = axes[2]
    sw = summary["c_ref_sweep"]
    cs = [r["c_ref"] for r in sw]
    for p in ("eq6", "eq6ramp", "exploit", "explore"):
        m = [r[p][0] for r in sw]
        e = [r[p][1] for r in sw]
        ax.errorbar(cs, m, yerr=e, marker="o", ms=6, lw=2.0, capsize=4,
                    color=colors[p], label=labels[p])
    ax.set_xlabel(r"cost of a reframe, $c_{\mathrm{ref}}$ (rounds)")
    ax.set_ylabel(r"$\mathbb{E}[Y_T]$")
    ax.set_title("(c) Sensitivity to reframing cost")
    ax.legend(loc="best", fontsize=10.5)

    fig.tight_layout()
    fig.savefig(fname)
    plt.close(fig)


if __name__ == "__main__":
    main()


# =========================================================================== #
#  G_ref MISSPECIFICATION STUDY  (Section 6.5)
#
#  Eq. (6) needs G_ref, the expected gain from entering a fresh frame, and that
#  is the one input the model cannot derive from inside the current frame.  A
#  criterion whose usefulness collapsed under a mildly wrong G_ref would not be
#  worth stating, so we measure the collapse.
#
#  We first CALIBRATE: run the loop to convergence in each frame of the pool and
#  record the outcome actually attained.  That empirical mean and s.d. define the
#  correctly specified prior.  We then displace the prior mean by +/-0.05 and
#  +/-0.10 utility (a large error: the whole spread of attainable outcomes is
#  about 0.1), shrink or inflate its width, and switch off learning altogether,
#  and re-measure the policy's outcome against the two naive baselines.
# =========================================================================== #
def calibrate_frame_outcomes(n_frames=None, n_iter=80, seed=99):
    """Empirical distribution of the outcome a team actually attains in a frame."""
    n_frames = n_frames or N_POOL
    vals = []
    for i in range(n_frames):
        rng = np.random.default_rng(seed * 7919 + i)
        f = Frame(_landscape(i), rng)
        for _ in range(n_iter):
            f.step()
        vals.append(f.best_true)
    vals = np.array(vals)
    return float(vals.mean()), float(vals.std(ddof=1)), vals


def gref_sensitivity(n_runs=300):
    m_true, sd_true, _ = calibrate_frame_outcomes()
    print("\n== G_ref misspecification study ==")
    print("  calibrated frame-outcome prior: mean %.4f, s.d. %.4f"
          % (m_true, sd_true))

    conds = [
        ("calibrated",            m_true,        sd_true, True),
        ("optimistic $+0.05$",    m_true + 0.05, sd_true, True),
        ("optimistic $+0.10$",    m_true + 0.10, sd_true, True),
        ("pessimistic $-0.05$",   m_true - 0.05, sd_true, True),
        ("pessimistic $-0.10$",   m_true - 0.10, sd_true, True),
        ("over-confident, $+0.05$ off", m_true + 0.05, 0.01, True),
        ("diffuse ($\\tau=0.15$)", m_true,       0.15,    True),
        ("no learning, $+0.05$ off",   m_true + 0.05, sd_true, False),
    ]
    rows = []
    for name, m0, tau, learn in conds:
        finals, nref = [], []
        for r in range(n_runs):
            d = run_policy("eq6", 7000 + r, prior_m0=m0, prior_tau2=tau ** 2,
                           learn_prior=learn)
            finals.append(d["final"]); nref.append(d["n_reframes"])
        finals = np.array(finals)
        rows.append({"name": name, "m0": m0, "tau": tau, "learn": learn,
                     "mean": float(finals.mean()),
                     "ci95": float(1.96 * finals.std(ddof=1) / np.sqrt(finals.size)),
                     "n_reframes": float(np.mean(nref))})
        print("  %-30s E[Y_T]=%.4f +/- %.4f   reframes/run %.2f"
              % (name, rows[-1]["mean"], rows[-1]["ci95"], rows[-1]["n_reframes"]))

    base = {}
    for pol in ("exploit", "explore"):
        v = np.array([run_policy(pol, 7000 + r)["final"] for r in range(n_runs)])
        base[pol] = {"mean": float(v.mean()),
                     "ci95": float(1.96 * v.std(ddof=1) / np.sqrt(v.size))}
        print("  %-30s E[Y_T]=%.4f +/- %.4f" % ("[baseline] " + pol,
                                                base[pol]["mean"], base[pol]["ci95"]))
    return {"m_true": m_true, "sd_true": sd_true, "rows": rows, "baselines": base}


def fig_gref(res, fname):
    rows = res["rows"]
    offs = [r["m0"] - res["m_true"] for r in rows[:5]]
    means = [r["mean"] for r in rows[:5]]
    cis = [r["ci95"] for r in rows[:5]]
    nref = [r["n_reframes"] for r in rows[:5]]
    order = np.argsort(offs)
    offs = np.array(offs)[order]; means = np.array(means)[order]
    cis = np.array(cis)[order]; nref = np.array(nref)[order]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.4, 4.6))
    a1.axhspan(res["baselines"]["exploit"]["mean"] - res["baselines"]["exploit"]["ci95"],
               res["baselines"]["exploit"]["mean"] + res["baselines"]["exploit"]["ci95"],
               color=C_EXPLOIT, alpha=0.18, lw=0)
    a1.axhline(res["baselines"]["exploit"]["mean"], color=C_EXPLOIT, ls="--", lw=1.6,
               label="never reframe (baseline)")
    a1.axhspan(res["baselines"]["explore"]["mean"] - res["baselines"]["explore"]["ci95"],
               res["baselines"]["explore"]["mean"] + res["baselines"]["explore"]["ci95"],
               color=C_EXPLORE, alpha=0.18, lw=0)
    a1.axhline(res["baselines"]["explore"]["mean"], color=C_EXPLORE, ls=":", lw=1.6,
               label="fixed schedule (baseline)")
    a1.errorbar(offs, means, yerr=cis, marker="o", ms=7, lw=2.2, capsize=4,
                color=C_EQ6, label="Eq. (6) with displaced prior")
    a1.axvline(0, color="#555", lw=0.9)
    a1.set_xlabel(r"error in the $G_{\mathrm{ref}}$ prior mean (utility)")
    a1.set_ylabel(r"$\mathbb{E}[Y_T]$")
    a1.set_title("(a) Outcome under a misspecified prior")
    a1.legend(fontsize=10.5, loc="lower center")

    a2.plot(offs, nref, marker="s", ms=7, lw=2.2, color=C_MATCH)
    a2.axvline(0, color="#555", lw=0.9)
    a2.set_xlabel(r"error in the $G_{\mathrm{ref}}$ prior mean (utility)")
    a2.set_ylabel("reframes per run")
    a2.set_title("(b) The mechanism: optimism buys reframes")
    fig.tight_layout()
    fig.savefig(fname)
    plt.close(fig)


def run_gref_study():
    res = gref_sensitivity()
    fig_gref(res, os.path.join(OUTDIR, "Fig14.png"))
    with open(os.path.join(OUTDIR, "gref_results.json"), "w") as f:
        json.dump(res, f, indent=2)
    print("\nwritten -> gref_results.json and Fig14.png")
    return res
