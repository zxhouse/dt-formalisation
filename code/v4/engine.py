#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shared engine for the follow-up studies (horizon, allocation, bias, baselines).

`loop()` is the Ideate-Prototype-Test loop of simulation.run_design_process,
generalised in one respect: the number of tests a round spends on NEW
prototypes (`n_new`), on re-testing the current leaders (`n_top`) and on the
round-robin re-test of older prototypes (`n_rr`) are separate arguments, so a
fixed per-round test budget can be split in different ways.  With
(n_new, n_top, n_rr) = (1, 3, 1), legacy=True and the same seed it reproduces
the reference implementation exactly (see `selfcheck`); the default differs from
the reference in one respect, the rotating re-test, which the reference
implements incorrectly (see the docstring of `loop`).

Prototypes are integer-coded (bit i = feature i), which lets every prototype be
looked up in the exhaustively enumerated landscape: the set of local optima is
known exactly, so "has this run reached a local optimum?" is a lookup, not an
inference.

Three outcome series are returned, and the distinction matters:
  shipped[k]  true utility of the prototype the team currently believes best
              (its posterior-mean incumbent) -- what it would ship at round k;
  retained[k] running maximum of `shipped` (the paper's Y_k);
  found[k]    best true utility among all prototypes tested so far, whether or
              not the team has recognised it;
  cautious[k] true utility of the prototype with the highest posterior mean
              minus one posterior standard deviation -- what a team would ship
              if it discounted thinly tested favourites;
  belief[k]   the posterior mean of the incumbent (the paper's X_k).

Dependencies: numpy, simulation.py, rates.py.
"""
import numpy as np

import simulation as S
import rates as R

N_FEATURES = 14


# --------------------------------------------------------------------------- #
#  Landscapes as lookup tables
# --------------------------------------------------------------------------- #
class Table:
    """A landscape reduced to its utility table plus exact local-optimum masks."""

    def __init__(self, U, name="", exact=None):
        self.U = np.asarray(U, dtype=float)
        self.n = int(np.log2(self.U.size))
        self.name = name
        self.ustar = float(self.U.max())
        self.gidx = int(np.argmax(self.U))
        self._exact = exact            # optional reference landscape object
        self._masks = {}

    def local_optima(self, r_max):
        if r_max not in self._masks:
            offs = R.hamming_ball_offsets(self.n, r_max)
            n_opt, mask = R.count_local_optima(self.U, offs)
            self._masks[r_max] = (int(n_opt), mask, int(offs.size))
        return self._masks[r_max]

    def fdc(self):
        codes = np.arange(self.U.size)
        dist = np.array([bin(int(c) ^ self.gidx).count("1") for c in codes])
        return float(np.corrcoef(self.U, dist)[0, 1])

    def utility_fn(self):
        """Per-prototype utility. Uses the reference object when one is given,
        so that trajectories match simulation.py bit for bit."""
        if self._exact is None:
            U = self.U
            return lambda c: float(U[c])
        land, n, cache = self._exact, self.n, {}

        def f(c):
            if c not in cache:
                cache[c] = land.utility(((c >> np.arange(n)) & 1).astype(float))
            return cache[c]
        return f


def dvf_table(landscape, name="dvf", exact=False):
    U, _ = R.enumerate_utilities(landscape)
    return Table(U, name=name, exact=landscape if exact else None)


def nk_table(K, seed, n=N_FEATURES):
    """Kauffman NK landscape: each feature's contribution depends on its own
    state and on K other features; utility is the mean contribution, in [0,1].
    K = 0 is additive (one optimum); ruggedness rises with K."""
    rng = np.random.default_rng(seed)
    contrib = rng.random((n, 1 << (K + 1)))
    nbrs = np.array([np.concatenate(([i], rng.choice(
        [j for j in range(n) if j != i], size=K, replace=False)))
        for i in range(n)])
    codes = np.arange(1 << n)
    bits = (codes[:, None] >> np.arange(n)[None, :]) & 1
    U = np.zeros(codes.size)
    for i in range(n):
        idx = np.zeros(codes.size, dtype=np.int64)
        for b, j in enumerate(nbrs[i]):
            idx |= bits[:, j] << b
        U += contrib[i, idx]
    return Table(U / n, name=f"NK(K={K})#{seed}")


# --------------------------------------------------------------------------- #
#  The loop
# --------------------------------------------------------------------------- #
def loop(tab, n_iter, n_new=1, n_top=3, n_rr=1, ideas=5, r_max=2,
         sigma=0.10, prior_var=0.08, bias=0.0, seed=0, memory=None,
         prior_mu=0.5, screen=True, legacy=False):
    """`memory`: if set, only the `memory` prototypes with the highest posterior
    mean are kept after each round and the rest are forgotten (a violation of
    the retention assumption); None keeps everything.
    `screen`: if True (reference behaviour) the idea to prototype next is the one
    that looks best on a rough pre-test estimate; if False it is drawn at random
    from the ideas proposed in the current round.
    `legacy`: reproduce simulation.run_design_process exactly, including its
    faulty "round-robin" re-test.  There the index k % len(prototypes) always
    lands on the prototype built in the current round, so nothing is rotated:
    each prototype gets two tests when built and older ones are never revisited
    unless they are among the leaders.  The default (legacy=False) uses a true
    rotation (a queue: the prototype at the front is re-tested and moved to the
    back), so every retained prototype is tested again and again, as the
    infinite re-test condition of the theorem requires.  In legacy mode the
    run also starts from `ideas` random ideas; otherwise always from five."""
    from collections import deque
    queue = deque()
    rng = np.random.default_rng(seed)
    n = tab.n
    ufun = tab.utility_fn()
    pow2 = (1 << np.arange(n)).astype(np.int64)

    beliefs, true_util, cand = {}, {}, {}

    def code(p):
        return int(p.astype(np.int64) @ pow2)

    def vec(c):
        return ((c >> np.arange(n)) & 1).astype(float)

    fresh = []

    def add(p):
        c = code(p)
        if c not in beliefs and c not in cand:
            tu = ufun(c)
            true_util[c] = tu
            cand[c] = tu + rng.normal(0, 0.10)       # cheap proxy for untested ideas
            fresh.append(c)

    for _ in range(ideas if legacy else 5):
        add((rng.random(n) < 0.3).astype(float))

    shipped = np.empty(n_iter)
    found = np.empty(n_iter)
    inc = np.empty(n_iter, dtype=np.int64)
    belief = np.empty(n_iter)
    cautious = np.empty(n_iter)
    fbest = np.empty(n_iter, dtype=np.int64)
    best_found, best_found_code = -1.0, -1
    tests = 0
    for k in range(n_iter):
        if k > 0:
            fresh.clear()
        # Ideate: bounded mutation of a current leader
        if beliefs:
            frontier = sorted(beliefs, key=lambda c: beliefs[c].mu,
                              reverse=True)[:max(1, n_top)]
            base = vec(frontier[rng.integers(len(frontier))])
        else:
            base = (rng.random(n) < 0.3).astype(float)
        for _ in range(ideas):
            child = base.copy()
            flips = rng.integers(1, r_max + 1)
            idx = rng.choice(n, size=flips, replace=False)
            child[idx] = 1.0 - child[idx]
            add(child)

        # Prototype + first test of n_new untested candidates
        for _ in range(n_new):
            if not cand:
                break
            if screen:
                ch = max(cand, key=cand.get)
            else:
                pool = [c for c in fresh if c in cand] or list(cand)
                ch = pool[rng.integers(len(pool))]
            del cand[ch]
            beliefs[ch] = S.UtilityBelief(prior_mu, prior_var, sigma)
            queue.append(ch)
            beliefs[ch].update(true_util[ch] + rng.normal(0, sigma))
            tests += 1
            if true_util[ch] > best_found:
                best_found, best_found_code = true_util[ch], ch

        # Re-test the leaders (confirmation bias, if any, favours the incumbent)
        if n_top > 0 and beliefs:
            top = sorted(beliefs, key=lambda c: beliefs[c].mu,
                         reverse=True)[:n_top]
            for rank, c in enumerate(top):
                b = bias if rank == 0 else 0.0
                beliefs[c].update(true_util[c] + b + rng.normal(0, sigma))
                tests += 1

        # Round-robin re-test, so that every retained prototype is revisited
        if n_rr > 0 and beliefs:
            keys = list(beliefs.keys())
            for j in range(n_rr):
                if legacy:
                    c = keys[(k * n_rr + j) % len(keys)]
                else:
                    c = queue.popleft()
                    while c not in beliefs:          # forgotten (memory limit)
                        c = queue.popleft()
                    queue.append(c)
                beliefs[c].update(true_util[c] + rng.normal(0, sigma))
                tests += 1

        if memory is not None and len(beliefs) > memory:
            keep = sorted(beliefs, key=lambda c: beliefs[c].mu,
                          reverse=True)[:memory]
            beliefs = {c: beliefs[c] for c in keep}
            queue = deque(c for c in queue if c in beliefs)

        leader = max(beliefs, key=lambda c: beliefs[c].mu)
        belief[k] = beliefs[leader].mu
        safe = max(beliefs, key=lambda c: beliefs[c].mu - np.sqrt(beliefs[c].var))
        cautious[k] = true_util[safe]
        shipped[k] = true_util[leader]
        inc[k] = leader
        found[k] = best_found
        fbest[k] = best_found_code
    return {"shipped": shipped, "retained": np.maximum.accumulate(shipped),
            "found": found, "belief": belief, "cautious": cautious, "incumbent": inc, "found_code": fbest,
            "tests": tests}


def selfcheck():
    land = S.UtilityLandscape()
    tab = dvf_table(land, exact=True)
    ref = S.run_design_process(land, n_iter=80, seed=1003)
    mine = loop(tab, 80, seed=1003, legacy=True)
    assert np.array_equal(ref["Xtrue"], mine["retained"]), "engine mismatch"
    assert np.array_equal(ref["best_true"], mine["shipped"]), "engine mismatch"
    return True


def main_table():
    return dvf_table(S.UtilityLandscape(), name="main", exact=True)


if __name__ == "__main__":
    print("selfcheck:", selfcheck())
    for K in (0, 2, 4, 8):
        t = nk_table(K, 1)
        print(t.name, "U*=%.3f" % t.ustar, "mean=%.3f sd=%.3f" % (t.U.mean(), t.U.std()),
              "optima r1/r2:", t.local_optima(1)[0], t.local_optima(2)[0],
              "FDC=%.2f" % t.fdc())
    m = main_table()
    print("main U*=%.4f sd=%.3f optima r1/r2:" % (m.ustar, m.U.std()),
          m.local_optima(1)[0], m.local_optima(2)[0], "FDC=%.2f" % m.fdc())
