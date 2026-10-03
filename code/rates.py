#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Finite-time / rate analysis for the Design-Thinking convergence model
(Section 5.5 of the paper, "How fast?").

Three quantities are computed exactly on the enumerable |F|=14 landscape:

  (R1) |N(p)|  -- the size of the operator-reachable neighbourhood (Hamming ball
       of radius r_max).  This, not |2^F|, is the effective arm count of the
       within-frame bandit, and it is what the finite-time bound scales with.

  (R2) L(r_max) -- the length of the longest STRICTLY IMPROVING path in the
       neighbourhood graph.  Because strict improvement is acyclic, the improving
       graph is a DAG and the longest path is computable in O(V+E) by processing
       vertices in increasing utility order.  L bounds the number of successful
       improvement events any run can require.

  (R3) delta_hat -- the empirically realised per-round generativity of the
       simulated Ideate/Prototype/Test operators: the fraction of rounds, taken
       over rounds in which the incumbent was NOT locally optimal, in which the
       retained-best true utility strictly increased.  This is the model's own
       delta, measured rather than assumed.

The finite-time proposition of the paper then predicts
      E[rounds to local optimality] <= L / delta,
which is checked against the observed convergence iteration.

Dependencies: numpy (+ simulation.py in the same directory).
"""
import json
import os
from math import comb

import numpy as np

import simulation as S

OUTDIR = os.path.dirname(os.path.abspath(__file__))
OUT_JSON = os.path.join(OUTDIR, "rates_results.json")


# --------------------------------------------------------------------------- #
#  Exhaustive enumeration of the landscape
# --------------------------------------------------------------------------- #
def enumerate_utilities(landscape):
    """Return U as a float array indexed by the integer encoding of p."""
    n = landscape.n
    N = 1 << n
    bits = ((np.arange(N)[:, None] >> np.arange(n)[None, :]) & 1).astype(float)
    return S._utility_batch(landscape, bits), bits


def hamming_ball_offsets(n, r_max):
    """All non-zero bit-masks of popcount <= r_max, as an int array."""
    offs = []
    idx = np.arange(n)
    for r in range(1, r_max + 1):
        # iterate over all r-subsets via bit tricks on a small n
        def rec(start, depth, acc):
            if depth == 0:
                offs.append(acc)
                return
            for i in range(start, n - depth + 1):
                rec(i + 1, depth - 1, acc | (1 << int(idx[i])))
        rec(0, r, 0)
    return np.array(sorted(set(offs)), dtype=np.int64)


def longest_improving_path(U, offsets):
    """
    Longest path (in edges) in the DAG of strictly improving moves
    q -> p with p in N(q) and U(p) > U(q).

    Processed in increasing utility order, so every predecessor of a vertex is
    already finalised: depth[p] = 1 + max over improving predecessors q.
    """
    N = U.shape[0]
    order = np.argsort(U, kind="stable")
    depth = np.zeros(N, dtype=np.int32)
    for p in order:
        up = U[p]
        nb = p ^ offsets                       # neighbours of p
        worse = nb[U[nb] < up]                 # improving predecessors q -> p
        if worse.size:
            depth[p] = depth[worse].max() + 1
    return int(depth.max()), depth


def count_local_optima(U, offsets):
    N = U.shape[0]
    allp = np.arange(N)
    best_nb = np.full(N, -np.inf)
    for off in offsets:
        best_nb = np.maximum(best_nb, U[allp ^ int(off)])
    return int(np.sum(U >= best_nb - 1e-15)), (U >= best_nb - 1e-15)


# --------------------------------------------------------------------------- #
#  (R3) measured generativity of the simulated operators
# --------------------------------------------------------------------------- #
def measure_delta(landscape, U, is_local_opt, n_runs=250, n_iter=80, **kw):
    """
    Rerun the loop, recording per round whether (a) the retained incumbent was
    N-locally optimal and (b) the retained-best TRUE utility strictly improved.
    delta_hat = P(improve | not locally optimal).
    """
    n = landscape.n
    pow2 = (1 << np.arange(n))

    opportunities = 0
    successes = 0
    first_hit = []          # round at which the run reaches its own limit
    hit99 = []              # round at which the run is within 1% of its limit
    n_improve = []          # realised length of the improving chain per run
    eps_stop = []           # epsilon-stopping iteration k* (Definition 1a proxy)

    for r in range(n_runs):
        out = S.run_design_process(landscape, n_iter=n_iter, seed=1000 + r, **kw)
        Y = out["Xtrue"]                    # retained-best true utility
        improved = np.diff(Y) > 1e-12
        # a round is an "opportunity" while the retained best is still below the
        # run's own final value (i.e. an improving move demonstrably existed)
        final = Y[-1]
        opp = Y[:-1] < final - 1e-12
        opportunities += int(opp.sum())
        successes += int((improved & opp).sum())
        n_improve.append(int(improved.sum()))
        first_hit.append(int(np.argmax(Y >= final - 1e-12)))
        hit99.append(int(np.argmax(Y >= 0.99 * final - 1e-12)))
        # epsilon-stopping: first k with Y[k+N]-Y[k] < eps  (N=5, eps=0.01)
        Nw, eps = 5, 0.01
        ks = n_iter - Nw - 1
        gain = Y[Nw:] - Y[:-Nw]
        idx = np.flatnonzero(gain < eps)
        eps_stop.append(int(idx[0]) if idx.size else ks)

    delta_hat = successes / max(1, opportunities)
    return dict(delta_hat=delta_hat,
                mean_first_hit=float(np.mean(first_hit)),
                median_first_hit=float(np.median(first_hit)),
                mean_hit99=float(np.mean(hit99)),
                median_hit99=float(np.median(hit99)),
                mean_improvements=float(np.mean(n_improve)),
                median_eps_stop=float(np.median(eps_stop)),
                mean_eps_stop=float(np.mean(eps_stop)))


def main():
    land = S.UtilityLandscape()
    U, bits = enumerate_utilities(land)
    res = {"n_features": land.n, "n_states": int(U.size),
           "U_max": float(U.max()), "U_min": float(U.min())}

    for r_max in (1, 2, 3):
        offs = hamming_ball_offsets(land.n, r_max)
        n_opt, mask = count_local_optima(U, offs)
        L, depth = longest_improving_path(U, offs)
        res[f"r{r_max}"] = {
            "neighbourhood_size": int(offs.size),
            "n_local_optima": n_opt,
            "longest_improving_path": L,
            "mean_depth": float(depth.mean()),
            "p90_depth": float(np.percentile(depth, 90)),
        }
        print(f"r_max={r_max}: |N|={offs.size:5d}  local optima={n_opt:3d}  "
              f"L={L:3d}  mean depth={depth.mean():.2f}")

    # measured generativity of the operators (baseline configuration)
    offs2 = hamming_ball_offsets(land.n, 2)
    _, mask2 = count_local_optima(U, offs2)
    meas = measure_delta(land, U, mask2)
    res.update(meas)
    L2 = res["r2"]["longest_improving_path"]
    delta_hat = meas["delta_hat"]
    res["bound_L_over_delta"] = L2 / delta_hat if delta_hat > 0 else None
    res["bound_realised_chain_over_delta"] = meas["mean_improvements"] / delta_hat
    print(f"\nmeasured delta_hat            = {delta_hat:.3f}")
    print(f"realised improving chain len  = {meas['mean_improvements']:.2f}")
    print(f"rounds to run-limit           : mean {meas['mean_first_hit']:.1f}, "
          f"median {meas['median_first_hit']:.0f}")
    print(f"rounds to within 1% of limit  : mean {meas['mean_hit99']:.1f}, "
          f"median {meas['median_hit99']:.0f}")
    print(f"eps-stopping iteration k*     : mean {meas['mean_eps_stop']:.1f}, "
          f"median {meas['median_eps_stop']:.0f}")
    print(f"worst-case bound  L/delta     = {L2}/{delta_hat:.3f} "
          f"= {L2/delta_hat:.0f} rounds  (valid but very loose)")
    print(f"realised-chain bound          = {meas['mean_improvements']:.2f}"
          f"/{delta_hat:.3f} = {meas['mean_improvements']/delta_hat:.1f} rounds")

    with open(OUT_JSON, "w") as f:
        json.dump(res, f, indent=2)
    print(f"\nwritten -> {OUT_JSON}")


if __name__ == "__main__":
    main()
