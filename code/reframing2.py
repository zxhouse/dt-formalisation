#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reframing follow-up: is the evidence-based criterion more robust than a
schedule?

reframing.py showed that, at one cost of reframing, the criterion beats never
reframing but only ties a fixed schedule.  A schedule, however, has a free
parameter (its period) that must be chosen in advance.  This script asks what
happens when the cost of a reframe varies and nothing is retuned: every fixed
period kappa is run at every cost, next to the criterion with three priors
(default, calibrated, optimistic).  All policies see the same held-out worlds
(paired seeds), so differences are paired.

Output: results/reframing2_results.json
Dependencies: numpy, reframing.py, simulation.py.  Fixed seeds.
"""
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

import reframing as RF

RESDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
C_REFS = [4, 10, 20, 35]
KAPPAS = [10, 15, 25, 40, 75]
M_CAL, SD_CAL = 0.7077, 0.0284          # calibrated frame-outcome prior
PRIORS = {"default": (None, None),
          "calibrated": (M_CAL, SD_CAL ** 2),
          "optimistic": (M_CAL + 0.05, SD_CAL ** 2)}
N_RUNS = 240
SEED0 = 20000


def _job(a):
    kind, par, c_ref, seed = a
    RF.C_REF_ROUNDS = c_ref
    if kind == "sched":
        RF.KAPPA = par
        d = RF.run_policy("explore", seed)
    elif kind == "never":
        d = RF.run_policy("exploit", seed)
    else:
        m0, t2 = PRIORS[par]
        d = RF.run_policy("eq6", seed, prior_m0=m0, prior_tau2=t2)
    return (kind, par, c_ref, seed, d["final"], d["n_reframes"])


def main():
    jobs = []
    for c in C_REFS:
        for s in range(N_RUNS):
            jobs.append(("never", 0, c, SEED0 + s))
            for k in KAPPAS:
                jobs.append(("sched", k, c, SEED0 + s))
            for p in PRIORS:
                jobs.append(("crit", p, c, SEED0 + s))
    t0 = time.time()
    with Pool(2) as pool:
        r = pool.map(_job, jobs, chunksize=8)
    print(f"{len(jobs)} jobs in {time.time()-t0:.0f}s", flush=True)

    cell = {}
    for kind, par, c, seed, y, n in r:
        cell.setdefault((kind, par, c), {})[seed] = (y, n)
    seeds = [SEED0 + s for s in range(N_RUNS)]

    def arr(key):
        return np.array([cell[key][s][0] for s in seeds])

    def ci(v):
        return float(1.96 * v.std(ddof=1) / np.sqrt(v.size))

    out = {"c_refs": C_REFS, "kappas": KAPPAS, "n_runs": N_RUNS, "rows": []}
    for c in C_REFS:
        row = {"c_ref": c}
        nv = arr(("never", 0, c))
        row["never"] = {"mean": float(nv.mean()), "ci95": ci(nv)}
        for k in KAPPAS:
            v = arr(("sched", k, c))
            row[f"sched{k}"] = {"mean": float(v.mean()), "ci95": ci(v),
                                "n_ref": float(np.mean([cell[("sched", k, c)][s][1] for s in seeds]))}
        for p in PRIORS:
            v = arr(("crit", p, c))
            row[f"crit_{p}"] = {"mean": float(v.mean()), "ci95": ci(v),
                                "n_ref": float(np.mean([cell[("crit", p, c)][s][1] for s in seeds])),
                                "minus_never": float((v - nv).mean()),
                                "minus_never_ci95": ci(v - nv)}
            for k in KAPPAS:
                d = v - arr(("sched", k, c))
                row[f"crit_{p}"][f"minus_sched{k}"] = float(d.mean())
                row[f"crit_{p}"][f"minus_sched{k}_ci95"] = ci(d)
        out["rows"].append(row)
        print(f"\nc_ref={c}: never {row['never']['mean']:.4f}")
        print("   schedule:", "  ".join(f"k={k}: {row[f'sched{k}']['mean']:.4f}" for k in KAPPAS))
        for p in PRIORS:
            x = row[f"crit_{p}"]
            print(f"   criterion[{p:10s}] {x['mean']:.4f} +/- {x['ci95']:.4f}  reframes {x['n_ref']:.2f}")

    # regret of each policy against the best policy at each cost, and its worst case
    names = ["never"] + [f"sched{k}" for k in KAPPAS] + [f"crit_{p}" for p in PRIORS]
    best = [max(row[n]["mean"] for n in names) for row in out["rows"]]
    out["regret"] = {n: [float(b - row[n]["mean"]) for b, row in zip(best, out["rows"])]
                     for n in names}
    print("\nshortfall against the best policy at each cost (x1000):")
    for n in names:
        g = out["regret"][n]
        print(f"   {n:18s}", "  ".join(f"{1000*x:5.1f}" for x in g),
              f"  worst {1000*max(g):5.1f}  mean {1000*np.mean(g):5.1f}")
    json.dump(out, open(os.path.join(RESDIR, "reframing2_results.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
