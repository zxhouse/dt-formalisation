#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reframing: schedules against the expected-improvement criterion.

A world holds six frames (independent DVF landscapes).  A team has T = 150
rounds, works in one frame at a time with the Ideate-Prototype-Test loop
(reframing.Frame: one new prototype and three re-tests of the leaders per
round), and may abandon the frame for a fresh one at a cost of c_ref rounds.
When the budget is spent it ships, among the favourites of the frames it has
visited, the one it believes to be best; the payoff is that prototype's TRUE
utility.  The team never sees true utilities.

Policies
  never       stay in the first frame
  every k     leave after k working rounds in a frame (k = 15, 25, 40, 75)
  criterion   leave when   EI - c_iter  <  G_ref - c_ref * c_iter
              after a minimum stay of DWELL rounds, where
              EI     expected improvement of one more round over the favourite's
                     belief: the larger of (a) the best rival among the tested
                     prototypes and (b) the most promising untested idea, judged
                     by its rough estimate shrunk towards the mean of the
                     team's beliefs in this frame (empirical Bayes);
              G_ref  expected improvement of a fresh frame over the best belief
                     the team holds, under a normal prior over what a frame
                     yields, updated with each frame the team leaves.
              Everything the criterion uses is a belief of the team.

This replaces the criterion policy of reframing.py / reframing2.py, whose EI
took an untested idea's rough estimate at face value with the diffuse prior
standard deviation, so that it never declined, and whose G_ref was given the
true utility of the team's favourite.

Output: results/reframing3_results.json.  Dependencies: numpy, reframing.py,
simulation.py.  Fixed seeds; all policies see the same worlds (paired).
"""
import json
import os
import time
from multiprocessing import Pool

import numpy as np

import reframing as RF

RESDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
T_BUDGET, M_FRAMES = 150, 6
C_ITER = 0.002                      # cost of one round, in utility units
DWELL = 10                          # minimum stay before the criterion may fire
PROXY_VAR = 0.10 ** 2               # variance of the rough estimate of an idea
SPREAD = 0.03                       # frame-to-frame spread of outcomes (prior)
C_REFS = (4, 10, 20, 35)
KAPPAS = (15, 25, 40, 75)
N_RUNS, SEED0 = 240, 30000
EI = RF.expected_improvement


def favourite(frame):
    k = max(frame.beliefs, key=lambda q: frame.beliefs[q].mu)
    b = frame.beliefs[k]
    return b.mu, frame.true_util[k]


def ei_within(frame):
    """Team's expected improvement from one more round in this frame."""
    bel = frame.beliefs
    fav = max(bel, key=lambda q: bel[q].mu)
    inc = bel[fav].mu
    best = 0.0
    rivals = sorted((q for q in bel if q != fav), key=lambda q: bel[q].mu,
                    reverse=True)[:3]
    for q in rivals:
        best = max(best, EI(bel[q].mu, np.sqrt(bel[q].var), inc))
    if frame.proxy:
        mus = np.array([b.mu for b in bel.values()])
        m_w, s_w = float(mus.mean()), max(float(mus.std()), 0.02)
        g = max(frame.proxy.values())
        v = 1.0 / (1.0 / PROXY_VAR + 1.0 / s_w ** 2)
        m = v * (g / PROXY_VAR + m_w / s_w ** 2)
        best = max(best, EI(m, np.sqrt(v), inc))
    return best


def run(policy, par, c_ref, seed, prior=(0.75, 0.05), trace=False):
    rng = np.random.default_rng(seed)
    RF.M_FRAMES = M_FRAMES
    world = RF.make_world(seed)
    order = rng.permutation(M_FRAMES)
    fi = 0
    frame = RF.Frame(world[order[fi]], rng)
    m, var = prior[0], prior[1] ** 2            # prior over what a frame yields
    held = []                                   # (belief, true) of favourites left behind
    oracle = 0.0
    t = t_frame = n_ref = 0
    when = []
    while t < T_BUDGET:
        frame.step()
        t += 1
        t_frame += 1
        mu, tru = favourite(frame)
        oracle = max(oracle, tru)
        can = fi + 1 < M_FRAMES and t + c_ref + DWELL <= T_BUDGET
        go = False
        if can and policy == "every":
            go = t_frame >= par
        elif can and policy == "criterion" and t_frame >= DWELL:
            y_hat = max([mu] + [h[0] for h in held])
            g_ref = EI(m, np.sqrt(var + SPREAD ** 2), y_hat)
            go = (ei_within(frame) - C_ITER) < (g_ref - c_ref * C_ITER)
        if go:
            held.append((mu, tru))
            p0, p1 = 1.0 / var, 1.0 / SPREAD ** 2      # learn what frames yield
            var = 1.0 / (p0 + p1)
            m = var * (p0 * m + p1 * mu)
            when.append(t)
            t += c_ref
            fi += 1
            n_ref += 1
            t_frame = 0
            frame = RF.Frame(world[order[fi]], rng)
    mu, tru = favourite(frame)
    held.append((mu, tru))
    shipped = max(held, key=lambda h: h[0])[1]
    out = {"shipped": shipped, "oracle": max(oracle, tru), "n_ref": n_ref}
    if trace:
        out["when"] = when
    return out


def calibrate(n=40, rounds=75):
    """What a frame yields: the belief about the favourite after `rounds` rounds."""
    v, tr = [], []
    for i in range(n):
        rng = np.random.default_rng(777 + i)
        f = RF.Frame(RF._landscape(i), rng)
        for _ in range(rounds):
            f.step()
        mu, tru = favourite(f)
        v.append(mu)
        tr.append(tru)
    return float(np.mean(v)), float(np.std(v, ddof=1)), float(np.mean(tr))


def _job(a):
    name, policy, par, c_ref, seed, prior = a
    d = run(policy, par, c_ref, seed, prior)
    return name, c_ref, seed, d["shipped"], d["oracle"], d["n_ref"]


def main():
    m_cal, s_cal, true_cal = calibrate()
    print(f"calibration: favourite's belief after 75 rounds {m_cal:.3f} (sd {s_cal:.3f}); "
          f"its true utility {true_cal:.3f}", flush=True)
    priors = {"default": (0.75, 0.05), "calibrated": (m_cal, max(s_cal, 0.02)),
              "optimistic": (m_cal + 0.05, max(s_cal, 0.02))}
    pols = [("never", "never", None, None)]
    pols += [(f"every{k}", "every", k, None) for k in KAPPAS]
    pols += [(f"crit_{p}", "criterion", None, priors[p]) for p in priors]
    jobs = [(n, pol, par, c, SEED0 + s, pr or (0.75, 0.05))
            for c in C_REFS for s in range(N_RUNS) for n, pol, par, pr in pols]
    t0 = time.time()
    with Pool(2) as pool:
        r = pool.map(_job, jobs, chunksize=16)
    print(f"{len(jobs)} jobs in {time.time()-t0:.0f}s", flush=True)

    cell = {}
    for name, c, seed, sh, orc, nr in r:
        cell.setdefault((name, c), {})[seed] = (sh, orc, nr)
    seeds = [SEED0 + s for s in range(N_RUNS)]
    names = [p[0] for p in pols]

    def col(name, c, j):
        return np.array([cell[(name, c)][s][j] for s in seeds])

    def ci(v):
        return float(1.96 * v.std(ddof=1) / np.sqrt(v.size))

    out = {"T": T_BUDGET, "frames": M_FRAMES, "c_iter": C_ITER, "dwell": DWELL,
           "n_runs": N_RUNS, "c_refs": list(C_REFS), "priors": priors,
           "calibration": {"belief_mean": m_cal, "belief_sd": s_cal, "true_mean": true_cal},
           "rows": []}
    for c in C_REFS:
        row = {"c_ref": c}
        nv = col("never", c, 0)
        for n in names:
            v = col(n, c, 0)
            row[n] = {"mean": float(v.mean()), "ci95": ci(v),
                      "oracle": float(col(n, c, 1).mean()),
                      "n_ref": float(col(n, c, 2).mean()),
                      "frac_reframing": float((col(n, c, 2) > 0).mean()),
                      "minus_never": float((v - nv).mean()), "minus_never_ci95": ci(v - nv),
                      "minus_every75": float((v - col("every75", c, 0)).mean()),
                      "minus_every75_ci95": ci(v - col("every75", c, 0))}
        out["rows"].append(row)
        print(f"\ncost {c}:")
        for n in names:
            x = row[n]
            print(f"   {n:16s} shipped {x['mean']:.4f} +/- {x['ci95']:.4f}   reframes {x['n_ref']:.2f}"
                  f"   vs never {x['minus_never']:+.4f} +/- {x['minus_never_ci95']:.4f}"
                  f"   vs once {x['minus_every75']:+.4f} +/- {x['minus_every75_ci95']:.4f}")
    # when does the criterion fire?
    w = []
    for s in range(120):
        w += run("criterion", None, 10, SEED0 + s, priors["default"], trace=True)["when"]
    w = np.array(w)
    if w.size:
        out["criterion_timing_cost10_default"] = {
            "n": int(w.size), "median_round": float(np.median(w)),
            "p10": float(np.percentile(w, 10)), "p90": float(np.percentile(w, 90))}
        print("\ncriterion (default prior, cost 10) fires at round: median %.0f, 10th-90th pct %.0f-%.0f"
              % (np.median(w), np.percentile(w, 10), np.percentile(w, 90)))
    # does the expected improvement of a further round decline along an epoch?
    tr = []
    for i in range(40):
        rng = np.random.default_rng(900 + i)
        f = RF.Frame(RF._landscape(i), rng)
        for k in range(75):
            f.step()
            if k >= 9:
                tr.append((k + 1, ei_within(f)))
    tr = np.array(tr)
    early = tr[tr[:, 0] <= 20, 1]
    late = tr[tr[:, 0] >= 40, 1]
    out["ei_along_epoch"] = {"mean_rounds_10_20": float(early.mean()),
                             "mean_rounds_40_75": float(late.mean()),
                             "frac_above_round_cost": float((tr[:, 1] > C_ITER).mean())}
    print("expected improvement of a further round: rounds 10-20 mean %.4f, rounds 40-75 mean %.4f; "
          "above the cost of a round in %.0f%% of rounds"
          % (early.mean(), late.mean(), 100 * (tr[:, 1] > C_ITER).mean()))
    json.dump(out, open(os.path.join(RESDIR, "reframing3_results.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
