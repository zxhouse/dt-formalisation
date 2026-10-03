#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Follow-up studies on the Ideate-Prototype-Test loop.

  horizon     Does a run actually reach a local optimum, and when?  Exact check
              against the enumerated set of local optima, up to 2,560 rounds.
  allocation  With a fixed number of tests per round, how should they be split
              between new prototypes (breadth) and re-tests (depth), and how
              does the answer move with test noise and with ideation volume?
  bias        Shape of the loss caused by confirmation bias.
  baselines   The loop against tuned search / bandit baselines at a matched
              test budget, tuned on one set of landscapes, scored on another.

All four run on three classes of landscape: the paper's main instance, a family
of 30 randomly generated desirability-viability-feasibility landscapes
(landscapes.draw_family), and Kauffman NK landscapes of varying ruggedness.

  dynamics    The 80-round baseline: belief, shipped, retained and found utility.
  ablation    One assumption relaxed at a time.

Usage:  python3 studies.py [horizon|allocation|bias|baselines|dynamics|ablation|all]
Outputs: results/<study>_results.json.   Dependencies: numpy.  Fixed seeds.
"""
import functools
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

import engine as E

OUTDIR = os.path.dirname(os.path.abspath(__file__))
RESDIR = os.path.join(OUTDIR, "results")
os.makedirs(RESDIR, exist_ok=True)
NPROC = max(1, os.cpu_count() or 1)


# --------------------------------------------------------------------------- #
#  Landscape registry (built lazily inside each worker process)
# --------------------------------------------------------------------------- #
@functools.lru_cache(maxsize=None)
def _dvf_family():
    import landscapes as L
    return L.draw_family(m=30)


@functools.lru_cache(maxsize=None)
def table(spec):
    kind = spec[0]
    if kind == "main":
        return E.main_table()
    if kind == "dvf":
        return E.dvf_table(_dvf_family()[spec[1]]["land"], name=f"DVF#{spec[1]}")
    if kind == "nk":
        return E.nk_table(spec[1], spec[2])
    raise ValueError(spec)


DVF_ALL = [("dvf", i) for i in range(30)]
NK_KS = (1, 2, 4, 6)
NK_ALL = [("nk", K, 200 + s) for K in NK_KS for s in range(5)]          # 20
NK_TUNE = [("nk", K, 100 + s) for K in NK_KS for s in range(2)]         # 8
MAIN = ("main",)


def ci95(x):
    x = np.asarray(x, dtype=float)
    return float(1.96 * x.std(ddof=1) / np.sqrt(x.size)) if x.size > 1 else 0.0


def pmap(fn, jobs, label=""):
    t0 = time.time()
    with Pool(NPROC) as pool:
        out = pool.map(fn, jobs, chunksize=max(1, len(jobs) // (NPROC * 8)))
    print(f"   [{label}] {len(jobs)} jobs in {time.time() - t0:.0f}s", flush=True)
    return out


# --------------------------------------------------------------------------- #
#  1. HORIZON
# --------------------------------------------------------------------------- #
CHECKS = [20, 40, 80, 160, 320, 640, 1280, 2560]


def _horizon_job(args):
    spec, r_max, seed, T = args
    tab = table(spec)
    _, is_opt, _ = tab.local_optima(r_max)
    out = E.loop(tab, T, r_max=r_max, seed=seed)
    rows = []
    for c in [c for c in CHECKS if c <= T]:
        i = c - 1
        rows.append((c,
                     out["shipped"][i] / tab.ustar,
                     out["retained"][i] / tab.ustar,
                     out["found"][i] / tab.ustar,
                     bool(is_opt[out["incumbent"][i]]),
                     bool(is_opt[out["found_code"][i]]),
                     bool(out["incumbent"][i] == tab.gidx),
                     out["cautious"][i] / tab.ustar))
    return spec, rows


def _agg_horizon(results):
    by = {}
    for _, rows in results:
        for (c, sh, rt, fd, inc_opt, found_opt, inc_glob, caut) in rows:
            by.setdefault(c, []).append((sh, rt, fd, inc_opt, found_opt, inc_glob, caut))
    out = {}
    for c, v in sorted(by.items()):
        a = np.array(v, dtype=float)
        out[str(c)] = {
            "shipped": float(a[:, 0].mean()), "shipped_ci95": ci95(a[:, 0]),
            "retained": float(a[:, 1].mean()),
            "found": float(a[:, 2].mean()),
            "incumbent_is_local_opt": float(a[:, 3].mean()),
            "best_tested_is_local_opt": float(a[:, 4].mean()),
            "incumbent_is_global_opt": float(a[:, 5].mean()),
            "cautious": float(a[:, 6].mean()), "cautious_ci95": ci95(a[:, 6]),
            "found_ci95": ci95(a[:, 2]),
            "n": int(a.shape[0])}
    return out


def horizon():
    print("== horizon ==", flush=True)
    res = {}
    main = table(MAIN)
    for r in (2, 1):
        n_opt, _, nb = main.local_optima(r)
        jobs = [(MAIN, r, 1000 + s, 2560) for s in range(120)]
        res[f"main_r{r}"] = {"n_local_optima": n_opt, "neighbourhood": nb,
                             "checkpoints": _agg_horizon(pmap(_horizon_job, jobs, f"main r={r}"))}
    fam = DVF_ALL[:10] + NK_ALL[::2]                       # 10 + 10 landscapes
    jobs = [(sp, 2, 3000 + s, 1280) for sp in fam for s in range(24)]
    r = pmap(_horizon_job, jobs, "families")
    res["dvf_family_r2"] = {"checkpoints": _agg_horizon([x for x in r if x[0][0] == "dvf"])}
    res["nk_family_r2"] = {"checkpoints": _agg_horizon([x for x in r if x[0][0] == "nk"])}
    for key in ("main_r2", "main_r1", "dvf_family_r2", "nk_family_r2"):
        print(f"\n  {key}", res[key].get("n_local_optima", ""))
        print("   rounds  shipped  cautious  found   incumbent-is-local-opt  best-tested-is-local-opt")
        for c, d in res[key]["checkpoints"].items():
            print(f"   {int(c):5d}   {d['shipped']*100:5.1f}%  {d['cautious']*100:5.1f}%  {d['found']*100:5.1f}%"
                  f"        {d['incumbent_is_local_opt']*100:5.1f}%"
                  f"                 {d['best_tested_is_local_opt']*100:5.1f}%")
    json.dump(res, open(os.path.join(RESDIR, "horizon_results.json"), "w"), indent=1)
    return res


# --------------------------------------------------------------------------- #
#  2. ALLOCATION  (breadth vs depth of testing, at a fixed budget)
# --------------------------------------------------------------------------- #
ALLOC_ROUNDS, ALLOC_BUDGET = 60, 5
SIGMAS = (0.025, 0.05, 0.10, 0.20)
IDEAS = (5, 15)


def _alloc_job(args):
    spec, n_new, sigma, ideas, seed = args
    tab = table(spec)
    out = E.loop(tab, ALLOC_ROUNDS, n_new=n_new, n_top=ALLOC_BUDGET - n_new,
                 n_rr=0, ideas=ideas, sigma=sigma, seed=seed)
    return (spec, n_new, sigma, ideas, out["shipped"][-1] / tab.ustar,
            out["found"][-1] / tab.ustar)


def allocation():
    print("== allocation ==", flush=True)
    sets = {"main": [(MAIN, 120)], "dvf": [(sp, 16) for sp in DVF_ALL],
            "nk": [(sp, 16) for sp in NK_ALL]}
    jobs = []
    for name, lst in sets.items():
        for sp, runs in lst:
            for n_new in range(1, ALLOC_BUDGET + 1):
                for sg in SIGMAS:
                    for ideas in IDEAS:
                        for s in range(runs):
                            jobs.append((sp, n_new, sg, ideas, 5000 + s))
    r = pmap(_alloc_job, jobs, "allocation")
    res = {"rounds": ALLOC_ROUNDS, "tests_per_round": ALLOC_BUDGET, "cells": {}}
    for name in sets:
        kind = {"main": "main", "dvf": "dvf", "nk": "nk"}[name]
        cell = {}
        for sg in SIGMAS:
            for ideas in IDEAS:
                # per-landscape means first, then mean and CI across landscapes
                per_new = {}
                for n_new in range(1, ALLOC_BUDGET + 1):
                    per_land = {}
                    for (sp, nn, s2, idn, shipped, found) in r:
                        if sp[0] == kind and nn == n_new and s2 == sg and idn == ideas:
                            per_land.setdefault(sp, []).append(shipped)
                    if kind == "main":
                        v = np.array(per_land[MAIN])
                        per_new[n_new] = {"mean": float(v.mean()), "ci95": ci95(v)}
                    else:
                        m = np.array([np.mean(x) for x in per_land.values()])
                        per_new[n_new] = {"mean": float(m.mean()), "ci95": ci95(m)}
                best = max(per_new, key=lambda k: per_new[k]["mean"])
                cell[f"sigma={sg},ideas={ideas}"] = {
                    "by_n_new": {str(k): v for k, v in per_new.items()},
                    "best_n_new": int(best),
                    "best_retest_share": 1 - best / ALLOC_BUDGET}
        res["cells"][name] = cell
    for name, cell in res["cells"].items():
        print(f"\n  {name}: shipped utility (% of U*) by number of NEW prototypes per round (of 5 tests)")
        for key, d in cell.items():
            row = "  ".join(f"{d['by_n_new'][str(k)]['mean']*100:5.2f}" for k in range(1, 6))
            print(f"   {key:22s} {row}   best: {d['best_n_new']} new")
    json.dump(res, open(os.path.join(RESDIR, "allocation_results.json"), "w"), indent=1)
    return res


# --------------------------------------------------------------------------- #
#  3. BIAS
# --------------------------------------------------------------------------- #
BIASES = (0.0, 0.025, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30)


BIAS_SIGMAS = (0.05, 0.10, 0.20)


def _bias_job(args):
    spec, b, seed, sigma = args
    tab = table(spec)
    out = E.loop(tab, 80, bias=b, seed=seed, sigma=sigma)
    return spec, b, out["shipped"][-1] / tab.ustar, out["retained"][-1] / tab.ustar, sigma


def bias():
    """Loss from a confirmation bias b on re-tests of the favourite, at three
    levels of test noise (the baseline, 0.10, is stored at the top level)."""
    print("== bias ==", flush=True)
    jobs = [(MAIN, b, 2000 + s, sg) for sg in BIAS_SIGMAS for b in BIASES for s in range(240)]
    jobs += [(sp, b, 2000 + s, sg) for sg in BIAS_SIGMAS for sp in DVF_ALL + NK_ALL
             for b in BIASES for s in range(16)]
    r = pmap(_bias_job, jobs, "bias")
    res = {"by_sigma": {}}
    for sg in BIAS_SIGMAS:
        block = {}
        for kind in ("main", "dvf", "nk"):
            rows = {}
            for b in BIASES:
                per_land = {}
                for sp, bb, sh, rt, s2 in r:
                    if sp[0] == kind and bb == b and s2 == sg:
                        per_land.setdefault(sp, []).append((sh, rt))
                if kind == "main":
                    a = np.array(per_land[MAIN])
                else:
                    a = np.array([np.mean(v, axis=0) for v in per_land.values()])
                rows[str(b)] = {"shipped": float(a[:, 0].mean()), "shipped_ci95": ci95(a[:, 0]),
                                "retained": float(a[:, 1].mean()), "retained_ci95": ci95(a[:, 1])}
            block[kind] = rows
            print(f"\n  sigma={sg}  {kind}:  bias -> shipped (% of U*), change from b=0")
            for b in BIASES:
                d = rows[str(b)]
                print(f"   b={b:5.3f}   {d['shipped']*100:6.2f} +/- {d['shipped_ci95']*100:4.2f}"
                      f"   {100*(d['shipped']-rows['0.0']['shipped']):+6.2f}")
        res["by_sigma"][str(sg)] = block
        if sg == 0.10:
            res.update(block)
    json.dump(res, open(os.path.join(RESDIR, "bias_results.json"), "w"), indent=1)
    return res


# --------------------------------------------------------------------------- #
#  4. BASELINES at a matched test budget
# --------------------------------------------------------------------------- #
BUDGET = 300


class Bench:
    """Shared harness: same oracle, same bounded mutation, same belief update."""

    def __init__(self, tab, seed, sigma=0.10, prior_var=0.08, r_max=2):
        import simulation as S
        self.S, self.tab, self.n = S, tab, tab.n
        self.U = tab.U
        self.sigma, self.pv, self.r_max = sigma, prior_var, r_max
        self.rng = np.random.default_rng(seed)
        self.b, self.cnt, self.tests = {}, {}, 0

    def rand(self):
        bits = (self.rng.random(self.n) < 0.3)
        return int(sum(1 << i for i in np.flatnonzero(bits)))

    def mutate(self, c):
        flips = self.rng.integers(1, self.r_max + 1)
        for i in self.rng.choice(self.n, size=flips, replace=False):
            c ^= (1 << int(i))
        return c

    def screened(self, c, ideas=5):
        """Best of `ideas` variants of c on the same rough pre-test estimate the
        loop uses (true utility plus N(0, 0.10) noise)."""
        cands = [self.mutate(c) for _ in range(ideas)]
        est = [self.U[q] + self.rng.normal(0, 0.10) for q in cands]
        return cands[int(np.argmax(est))]

    def test(self, c):
        if c not in self.b:
            self.b[c] = self.S.UtilityBelief(0.5, self.pv, self.sigma)
            self.cnt[c] = 0
        self.b[c].update(self.U[c] + self.rng.normal(0, self.sigma))
        self.cnt[c] += 1
        self.tests += 1

    def leader(self):
        return max(self.b, key=lambda c: self.b[c].mu)

    def left(self):
        return self.tests < BUDGET

    def shipped(self):
        return float(self.U[self.leader()] / self.tab.ustar)


def a_random(tab, seed, _):
    h = Bench(tab, seed)
    pool = [h.rand() for _ in range(5)]
    for c in pool:
        h.test(c)
    while h.left():
        c = h.mutate(pool[h.rng.integers(len(pool))])
        h.test(c)
        pool.append(c)
    return h.shipped()


def a_greedy(tab, seed, reps):
    """Noisy hill-climbing: test each 1-flip neighbour `reps` times, move to the
    best one that beats the current point; restart at random when stuck."""
    h = Bench(tab, seed)
    cur = h.rand()
    for _ in range(reps):
        h.test(cur)
    while h.left():
        best, best_mu = None, h.b[cur].mu
        for i in h.rng.permutation(h.n):
            q = cur ^ (1 << int(i))
            for _ in range(reps):
                if h.left():
                    h.test(q)
            if q in h.b and h.b[q].mu > best_mu:
                best, best_mu = q, h.b[q].mu
            if not h.left():
                break
        if best is None:
            cur = h.rand()
            if h.left():
                h.test(cur)
        else:
            cur = best
    return h.shipped()


def a_eps(tab, seed, eps):
    h = Bench(tab, seed)
    for _ in range(5):
        h.test(h.rand())
    while h.left():
        lead = h.leader()
        h.test(h.mutate(lead) if h.rng.random() < eps else lead)
    return h.shipped()


def a_ucb(tab, seed, par):
    c_explore, add_every = par
    h = Bench(tab, seed)
    for _ in range(5):
        h.test(h.rand())
    step = 0
    while h.left():
        step += 1
        if step % add_every == 0:
            h.test(h.mutate(h.leader()))
            continue
        lt = np.log(max(2, h.tests))
        pick = max(h.b, key=lambda c: h.b[c].mu + c_explore * np.sqrt(lt / h.cnt[c]))
        h.test(pick)
    return h.shipped()


def a_ucb_screen(tab, seed, par):
    """UCB whose new variants are pre-screened exactly as the loop's are."""
    c_explore, add_every = par
    h = Bench(tab, seed)
    for _ in range(5):
        h.test(h.rand())
    step = 0
    while h.left():
        step += 1
        if step % add_every == 0:
            h.test(h.screened(h.leader()))
            continue
        lt = np.log(max(2, h.tests))
        pick = max(h.b, key=lambda c: h.b[c].mu + c_explore * np.sqrt(lt / h.cnt[c]))
        h.test(pick)
    return h.shipped()


def a_thompson(tab, seed, add_every):
    h = Bench(tab, seed)
    for _ in range(5):
        h.test(h.rand())
    step = 0
    while h.left():
        step += 1
        if step % add_every == 0:
            h.test(h.mutate(h.leader()))
            continue
        keys = list(h.b.keys())
        mu = np.array([h.b[c].mu for c in keys])
        sd = np.sqrt(np.array([h.b[c].var for c in keys]))
        h.test(keys[int(np.argmax(mu + sd * h.rng.standard_normal(len(keys))))])
    return h.shipped()


def a_loop(tab, seed, _):
    out = E.loop(tab, BUDGET // 5, seed=seed)       # 1 new + 3 leaders + 1 round-robin
    return float(out["shipped"][-1] / tab.ustar)


def a_loop_blind(tab, seed, _):
    out = E.loop(tab, BUDGET // 5, seed=seed, screen=False)
    return float(out["shipped"][-1] / tab.ustar)


UCB_C = (0.0125, 0.025, 0.05, 0.1, 0.2, 0.4)

ALGOS = {
    "random search": (a_random, [None]),
    "greedy hill-climbing": (a_greedy, [1, 2, 4, 6, 8]),
    "epsilon-greedy": (a_eps, [0.1, 0.2, 0.3, 0.5, 0.7, 0.9]),
    "UCB": (a_ucb, [(c, a) for c in UCB_C for a in (2, 3, 5)]),
    "Thompson sampling": (a_thompson, [2, 3, 5]),
    "UCB with pre-screening": (a_ucb_screen, [(c, a) for c in UCB_C for a in (2, 3, 5)]),
    "I-P-T loop, no pre-screening": (a_loop_blind, [None]),
    "I-P-T loop": (a_loop, [None]),
}


def _bench_job(args):
    name, par, spec, seed = args
    return name, par, spec, ALGOS[name][0](table(spec), seed, par)


def baselines():
    print("== baselines ==", flush=True)
    tune_set = DVF_ALL[:10] + NK_TUNE
    eval_sets = {"main": [(MAIN, 250)],
                 "dvf": [(sp, 30) for sp in DVF_ALL[10:]],
                 "nk": [(sp, 30) for sp in NK_ALL]}
    # --- tuning (separate landscapes and seeds) ---
    jobs = [(name, par, sp, 9000 + s) for name, (_, grid) in ALGOS.items()
            for par in grid if len(grid) > 1 for sp in tune_set for s in range(10)]
    r = pmap(_bench_job, jobs, "tuning")
    chosen, tuning = {}, {}
    for name, (_, grid) in ALGOS.items():
        if len(grid) == 1:
            chosen[name] = grid[0]
            continue
        score = {}
        for par in grid:
            per_land = {}
            for n2, p2, sp, v in r:
                if n2 == name and p2 == par:
                    per_land.setdefault(sp, []).append(v)
            score[str(par)] = float(np.mean([np.mean(x) for x in per_land.values()]))
        best = max(grid, key=lambda p: score[str(p)])
        chosen[name], tuning[name] = best, score
        print(f"   tuned {name:22s} -> {best}   ({score})")
    # --- evaluation on held-out landscapes / seeds ---
    jobs = [(name, chosen[name], sp, 12000 + s) for name in ALGOS
            for lst in eval_sets.values() for sp, runs in lst for s in range(runs)]
    r = pmap(_bench_job, jobs, "evaluation")
    res = {"budget": BUDGET, "chosen": {k: (list(v) if isinstance(v, tuple) else v)
                                         for k, v in chosen.items()},
           "tuning_scores": tuning, "eval": {}}
    for setname, lst in eval_sets.items():
        kind = lst[0][0][0]
        per = {name: {} for name in ALGOS}
        for name, par, sp, v in r:
            if sp[0] == kind:
                per[name].setdefault(sp, []).append(v)
        block = {}
        loop_means = {sp: float(np.mean(v)) for sp, v in per["I-P-T loop"].items()}
        for name in ALGOS:
            if kind == "main":
                v = np.array(per[name][MAIN]); lv = np.array(per["I-P-T loop"][MAIN])
                sp_ = np.sqrt((v.var(ddof=1) + lv.var(ddof=1)) / 2)
                block[name] = {"mean": float(v.mean()), "ci95": ci95(v),
                               "mcse": float(v.std(ddof=1) / np.sqrt(v.size)),
                               "d_vs_loop": float((v.mean() - lv.mean()) / sp_) if name != "I-P-T loop" else 0.0}
            else:
                m = {sp: float(np.mean(x)) for sp, x in per[name].items()}
                diff = np.array([m[sp] - loop_means[sp] for sp in m])
                block[name] = {"mean": float(np.mean(list(m.values()))),
                               "ci95": ci95(list(m.values())),
                               "diff_vs_loop": float(diff.mean()), "diff_ci95": ci95(diff),
                               "frac_landscapes_beating_loop": float((diff > 0).mean()),
                               "n_landscapes": len(m)}
        res["eval"][setname] = block
        print(f"\n  {setname}: shipped utility, % of U*")
        for name, d in block.items():
            extra = (f"d={d['d_vs_loop']:+.2f}" if kind == "main" else
                     f"diff {d['diff_vs_loop']*100:+.2f} +/- {d['diff_ci95']*100:.2f}, "
                     f"beats loop on {d['frac_landscapes_beating_loop']*100:.0f}%")
            print(f"   {name:22s} {d['mean']*100:6.2f} +/- {d['ci95']*100:4.2f}   {extra}")
    json.dump(res, open(os.path.join(RESDIR, "baselines_results.json"), "w"), indent=1)
    return res


# --------------------------------------------------------------------------- #
#  5. DYNAMICS  (the 80-round baseline: belief, shipped, retained, found)
# --------------------------------------------------------------------------- #
def _dyn_job(args):
    seed, calibrated = args
    tab = table(MAIN)
    kw = {}
    if calibrated:          # prior matched to the landscape's utility distribution
        kw = {"prior_mu": float(tab.U.mean()), "prior_var": float(tab.U.var())}
    o = E.loop(tab, 80, seed=seed, **kw)
    return (o["belief"], o["shipped"], o["retained"], o["found"])


def dynamics():
    print("== dynamics ==", flush=True)
    r = pmap(_dyn_job, [(1000 + s, False) for s in range(250)], "dynamics")
    rc = np.array(pmap(_dyn_job, [(1000 + s, True) for s in range(250)], "calibrated"))
    ustar = table(MAIN).ustar
    a = np.array(r)                                   # runs x 4 x rounds
    res = {"U_star": ustar, "n_runs": int(a.shape[0]), "rounds": int(a.shape[2])}
    for i, name in enumerate(("belief", "shipped", "retained", "found")):
        res[name] = {"mean": a[:, i].mean(axis=0).tolist(),
                     "ci95": (1.96 * a[:, i].std(axis=0, ddof=1) / np.sqrt(a.shape[0])).tolist()}
    d = np.diff(a[:, 0], axis=1)                      # increments of the belief maximum
    res["belief_increment"] = {"mean": d.mean(axis=0).tolist(),
                               "ci95": (1.96 * d.std(axis=0, ddof=1) / np.sqrt(d.shape[0])).tolist()}
    lo = d.mean(axis=0) - 1.96 * d.std(axis=0, ddof=1) / np.sqrt(d.shape[0])
    res["frac_rounds_increment_ci_above_zero_or_covering"] = float((d.mean(axis=0) + 1.96 * d.std(axis=0, ddof=1) / np.sqrt(d.shape[0]) >= 0).mean())
    res["frac_rounds_mean_increment_nonneg"] = float((d.mean(axis=0) >= 0).mean())
    res["curse_at_80"] = float((a[:, 0, -1] - a[:, 1, -1]).mean())
    res["curse_at_80_ci95"] = ci95(a[:, 0, -1] - a[:, 1, -1])
    res["curse_at_5"] = float((a[:, 0, 4] - a[:, 1, 4]).mean())
    res["calibrated_prior"] = {
        "prior_mu": float(table(MAIN).U.mean()), "prior_sd": float(table(MAIN).U.std()),
        "belief": float(rc[:, 0, -1].mean()), "shipped": float(rc[:, 1, -1].mean()),
        "shipped_ci95": ci95(rc[:, 1, -1]), "found": float(rc[:, 3, -1].mean()),
        "curse_at_80": float((rc[:, 0, -1] - rc[:, 1, -1]).mean()),
        "curse_at_80_ci95": ci95(rc[:, 0, -1] - rc[:, 1, -1]),
        "belief_mean": rc[:, 0].mean(axis=0).tolist()}
    res["shipped_ci95_at_80"] = ci95(a[:, 1, -1])
    print("   calibrated prior: shipped %.2f%% of U*, belief %.2f%%, curse %.4f +/- %.4f"
          % (100 * res["calibrated_prior"]["shipped"] / ustar,
             100 * res["calibrated_prior"]["belief"] / ustar,
             res["calibrated_prior"]["curse_at_80"], res["calibrated_prior"]["curse_at_80_ci95"]))
    for name in ("belief", "shipped", "retained", "found"):
        print(f"   {name:9s} at 80 rounds: {res[name]['mean'][-1]/ustar*100:.2f}% of U*")
    print("   belief minus shipped (optimizer's curse): round 5 %.4f, round 80 %.4f +/- %.4f"
          % (res["curse_at_5"], res["curse_at_80"], res["curse_at_80_ci95"]))
    print("   rounds with non-negative mean belief increment: %.0f%%; with CI not below zero: %.0f%%"
          % (100 * res["frac_rounds_mean_increment_nonneg"],
             100 * res["frac_rounds_increment_ci_above_zero_or_covering"]))
    json.dump(res, open(os.path.join(RESDIR, "dynamics_results.json"), "w"), indent=1)
    return res


# --------------------------------------------------------------------------- #
#  6. ABLATION  (one assumption relaxed at a time, 80 rounds)
# --------------------------------------------------------------------------- #
ABL = [("baseline", {}),
       ("radius 1", {"r_max": 1}), ("radius 3", {"r_max": 3}),
       ("radius 14 (unbounded)", {"r_max": 14}),
       ("1 idea per round", {"ideas": 1}), ("2 ideas per round", {"ideas": 2}),
       ("15 ideas per round", {"ideas": 15}),
       ("memory 1 (no retention)", {"memory": 1}), ("memory 3", {"memory": 3}),
       ("memory 10", {"memory": 10}),
       ("no re-testing", {"n_top": 0, "n_rr": 0}),
       ("leaders only (no round-robin)", {"n_rr": 0}),
       ("no pre-screening of ideas", {"screen": False}),
       ("noise 0.05", {"sigma": 0.05}), ("noise 0.20", {"sigma": 0.20})]


def _abl_job(args):
    spec, i, seed = args
    tab = table(spec)
    o = E.loop(tab, 80, seed=seed, **ABL[i][1])
    return (spec, i, o["shipped"][-1] / tab.ustar, o["found"][-1] / tab.ustar,
            o["belief"][-1] - o["shipped"][-1])


def ablation():
    print("== ablation ==", flush=True)
    sets = {"main": [(MAIN, 200)], "dvf": [(sp, 16) for sp in DVF_ALL],
            "nk": [(sp, 16) for sp in NK_ALL]}
    jobs = [(sp, i, 7000 + s) for lst in sets.values() for sp, runs in lst
            for i in range(len(ABL)) for s in range(runs)]
    r = pmap(_abl_job, jobs, "ablation")
    res = {"rounds": 80, "conditions": [a[0] for a in ABL], "sets": {}}
    for name in sets:
        rows = {}
        for i, (label, _) in enumerate(ABL):
            per = {}
            for (sp, j, sh, fd, curse) in r:
                if sp[0] == name and j == i:
                    per.setdefault(sp, []).append((sh, fd, curse))
            if name == "main":
                a = np.array(per[MAIN])
            else:                                     # landscape means first
                a = np.array([np.mean(v, axis=0) for v in per.values()])
            rows[label] = {"shipped": float(a[:, 0].mean()), "shipped_ci95": ci95(a[:, 0]),
                           "found": float(a[:, 1].mean()), "found_ci95": ci95(a[:, 1]),
                           "curse": float(a[:, 2].mean())}
        res["sets"][name] = rows
    print("\n   condition                       main: shipped found | dvf: shipped found | nk: shipped found   (% of U*)")
    for label in res["conditions"]:
        print("   %-30s" % label + " | ".join(
            "  %6.2f %6.2f" % (100 * res["sets"][n][label]["shipped"],
                               100 * res["sets"][n][label]["found"]) for n in sets))
    json.dump(res, open(os.path.join(RESDIR, "ablation_results.json"), "w"), indent=1)
    return res


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    assert E.selfcheck()
    todo = {"horizon": horizon, "allocation": allocation, "bias": bias,
            "baselines": baselines, "dynamics": dynamics,
            "ablation": ablation}
    for name, fn in todo.items():
        if which in (name, "all"):
            fn()
