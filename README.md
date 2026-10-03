# When Does Design Thinking Work? A Convergence Theorem for the Iterate–Test Core

Companion code and reproduction package for the paper

> **When Does Design Thinking Work? A Convergence Theorem for the Iterate–Test Core**
> Paweł Kuraś, Adrian Michalski, Alicja Gerka, Patryk Organisciak

This repository contains everything needed to reproduce the figures, tables and
reported statistics of the paper: a Monte-Carlo simulation of the
Ideate → Prototype → Test loop, the landscape-geometry and rate analysis, the
reframing-policy and misspecification studies, a landscape-family robustness
study, a matched-budget benchmark against baseline algorithms, and the coded
protocol demonstrations. All experiments use fixed random seeds and depend only
on `numpy` and `matplotlib`, so every number is reproducible end-to-end.

---

## What the model is (and is not)

The paper analyses a **normative, idealized** model of the *decision core* of
Design Thinking within a single problem-framing epoch. The mathematics *proves*
conditional almost-sure convergence of the retained-best expected utility to a
local optimum; the simulation *demonstrates* internal consistency of the model
and maps the parameter space; whether real teams satisfy the assumptions
*remains untested* (a pre-registered validation protocol is described in the
paper). Nothing in this repository is empirical evidence about design teams —
the Monte-Carlo study checks that the *implemented model* behaves as proved, not
that Design Thinking does.

---

## Repository layout

```
design-thinking-formalization/
├── README.md
├── LICENSE                 # MIT
├── CITATION.cff
├── requirements.txt        # numpy, matplotlib
├── run_all.sh              # one-command clean-room reproduction
├── code/                   # all experiment scripts
│   ├── simulation.py       # I–P–T loop, ensemble, core figures
│   ├── analysis.py         # sensitivity + ablation sweep
│   ├── rates.py            # landscape geometry & realized rates
│   ├── reframing.py        # reframing-policy comparison + G_ref study
│   ├── landscapes.py       # landscape-family robustness study
│   ├── benchmarks.py       # matched-budget baseline comparison
│   ├── empirical.py        # coded published cases + coder-perturbation
│   ├── protocol_demo.py    # synthetic coded-protocol dry run
│   └── diagrams.py         # process-flow and state-space diagrams
├── figures/                # Fig1–Fig16 (PNG) as used in the paper
├── results/                # raw numerical outputs (JSON) + generated table rows
```

## Requirements

- Python ≥ 3.9
- `numpy`, `matplotlib` (`pip install -r requirements.txt`)

No other dependencies (no `scipy`, `networkx`, or GPU).

## Reproducing everything

```bash
pip install -r requirements.txt
bash run_all.sh
```

`run_all.sh` deletes any cached numerical outputs, re-runs the whole pipeline
from scratch in dependency order, and collects the regenerated figures into
`figures/` and the numerical outputs into `results/`. Runtime is roughly
30–45 minutes on a typical laptop; `landscapes.py` and `benchmarks.py` are the
two heavy stages. Individual scripts can also be run directly, e.g.
`cd code && python3 simulation.py`.

## Which script produces which figure

| Figure | Script | What it shows |
|--------|--------|---------------|
| Fig1, Fig2 | `diagrams.py` | DT process flow; state-space graph |
| Fig3 | `simulation.py` | Bayesian update / posterior concentration |
| Fig4 | `simulation.py` | Utility landscape & value histogram |
| Fig5 | `simulation.py` | Submartingale dynamics (theorem-consistent) |
| Fig6 | `simulation.py` | ε-stopping rule |
| Fig8 | `simulation.py` | Entropy across divergent/convergent phases |
| Fig9 | `simulation.py` | Convergence diagnostics (Lyapunov potential, k\*) |
| Fig10 | `analysis.py` | Sensitivity & ablation |
| Fig11 | `protocol_demo.py` | Synthetic coded-protocol dry run |
| Fig12 | `empirical.py` | Feasibility run on two published cases |
| Fig13 | `reframing.py` | Reframing-policy comparison |
| Fig14 | `reframing.py` (`run_gref_study`) | G_ref misspecification study |
| Fig15 | `landscapes.py` | Robustness across a family of landscapes |
| Fig16 | `benchmarks.py` | Matched-budget baseline comparison |

Tables: `analysis.py` writes `results/sensitivity_rows.tex` and
`results/ablation_rows.tex` (the bodies of the sensitivity and ablation tables);
`rates.py`, `reframing.py`, `landscapes.py` and `benchmarks.py` write their
numerical results to the corresponding `results/*.json`.

## Key reproduced numbers (fixed seeds)

| Quantity | Value |
|----------|-------|
| Global optimum U\* (main landscape, exhaustive) | 0.7634 |
| Ensemble-mean X∞ (baseline, 250 runs) | 0.7373 |
| Runs converging to a strict *local* optimum | 96.8% |
| Cumulative drift E[Δ] ± 95% CI | +0.065 ± 0.007 |
| Local optima at r_max = 1 / 2 / 3 | 4 / 2 / 1 |
| Measured generativity δ̂ | 0.104 |
| Rounds to run-limit / to within 1% / ε-stop k\* | 46.1 / 31.8 / 2.5 |
| Landscape family (M=30): mean attained fraction of U\* | 97.7% |
| Benchmarks (300-test budget), DT-loop shipped E[Y] | 0.7238 |
| ε-greedy / UCB / random / greedy shipped E[Y] | 0.7177 / 0.7144 / 0.6944 / 0.6779 |

## Notes on the model implementation

Two implementation choices make the simulation match the *idealized* process the
theorem is about rather than merely resemble it:

- **Bounded belief (projection onto [0,1]).** The theorem's posterior mean
  `M_k^p = E[U(p)|F_k]` is a conditional expectation of a bounded utility and so
  lies in [0,1]. The conjugate normal–normal update is an unbounded Gaussian
  approximation to it, so `UtilityBelief.update` projects the posterior mean onto
  [0,1]. The projection is almost never active and changes no reported number.
- **Round-robin exploratory retest.** Assumption 4 requires every retained
  prototype to be tested infinitely often. Because a top-r retest rule alone can
  let an older prototype fall out of the leading set, `run_design_process` adds
  one round-robin retest per round, so the implemented loop genuinely satisfies
  the infinite-retest condition over an infinite horizon.

## Reproducibility and seeds

Every script sets an explicit `numpy` random seed; ensemble runs derive per-run
seeds deterministically. Re-running on the same Python/`numpy` version yields the
numbers in the paper up to Monte-Carlo rounding. `run_all.sh` performs a full
clean-room reproduction (it removes cached outputs first).

## Declaration of generative-AI use

A large-language-model assistant helped draft/revise prose and helped write and
debug the accompanying Python code and reference formatting. It was **not** used
to originate the theoretical model, assumptions, proofs, choice of experiments,
or interpretation of results, and was not used to generate or alter any data.
The commit history of this repository preserves the record of code changes so
that implementation support remains distinguishable from the conceptual and
mathematical contributions.

## License

Code is released under the MIT License (see `LICENSE`). Please cite the paper if
you use this code (see `CITATION.cff`).

criterion uses the team's beliefs only, has a minimum stay,
  and the payoff is the true utility of the prototype the team would ship.
