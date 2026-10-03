#!/usr/bin/env bash
# =============================================================================
# Clean-room reproduction driver.
#
# Regenerates every figure, table row and reported statistic in the paper from
# scratch, using fixed random seeds.  Runs the scripts in dependency order and
# collects outputs into figures/ and results/.
#
# Usage:   bash run_all.sh
# Deps:    python3 with numpy + matplotlib   (pip install -r requirements.txt)
# Runtime: ~30-45 min on a typical laptop (landscapes.py and benchmarks.py are
#          the two heavy stages; everything else is a few minutes total).
# =============================================================================
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE/code"

# analysis.py runs under a soft per-invocation time budget; give it plenty.
export MAXTIME="${MAXTIME:-3000}"

# Remove any cached numerical outputs so everything recomputes from scratch.
rm -f analysis_results.json rates_results.json reframing_results.json \
      gref_results.json landscapes_results.json benchmarks_results.json

run_v4() {
  echo ">>> studies.py all (v4: horizon, allocation, bias, baselines, dynamics, ablation)"
  python3 studies.py all
  echo ">>> reframing2.py (v4: reframing schedules vs criterion)"; python3 reframing2.py
  echo ">>> figures_v4.py (v4: FigA-FigD)";                        python3 figures_v4.py
  mkdir -p "$HERE/figures" "$HERE/results"
  mv -f Fig[A-D].png "$HERE/figures/" 2>/dev/null || true
  mv -f results/*_results.json "$HERE/results/" 2>/dev/null || true
}
run_v5() {
  echo ">>> studies.py all (v5: horizon, allocation, bias, baselines, dynamics, ablation)"
  python3 studies.py all
  echo ">>> reframing3.py (v5: reframing schedules vs criterion)"; python3 reframing3.py
  echo ">>> figures_v5.py (v5: FigA_v5-FigD_v5)";                  python3 figures_v5.py
  echo ">>> fig1_v6.py (Figure 1)";                                python3 fig1_v6.py
  mkdir -p "$HERE/figures" "$HERE/results"
  mv -f Fig[A-D]_v5.png Fig1_v6.png "$HERE/figures/" 2>/dev/null || true
  mv -f results/*_results.json "$HERE/results/" 2>/dev/null || true
}
# `bash run_all.sh v5` runs only the studies of the current (v5) paper.
if [ "$1" = "v5" ]; then run_v5; exit 0; fi
# `bash run_all.sh v4` runs the v4 studies from code/v4 (kept for the record).
if [ "$1" = "v4" ]; then cd "$HERE/code/v4"; cp -n ../simulation.py ../rates.py ../landscapes.py ../reframing.py . 2>/dev/null || true; run_v4; exit 0; fi


echo "==== clean-room reproduction started $(date) ===="
echo ">>> diagrams.py (Fig1, Fig2)";        python3 diagrams.py
echo ">>> simulation.py (Fig3-6, Fig8-9)";  python3 simulation.py
echo ">>> rates.py (Table: rates)";         python3 rates.py
echo ">>> analysis.py (Fig10, sens/abl)";   python3 analysis.py
echo ">>> reframing.py (Fig13, Fig14)";     python3 -c "import reframing; reframing.main(); reframing.run_gref_study()"
echo ">>> empirical.py (Fig12)";            python3 empirical.py
echo ">>> protocol_demo.py (Fig11)";        python3 protocol_demo.py
echo ">>> landscapes.py (Fig15)";           python3 landscapes.py
echo ">>> benchmarks.py (Fig16)";           python3 benchmarks.py

run_v5

# Collect outputs.
mkdir -p "$HERE/figures" "$HERE/results"
mv -f Fig*.png "$HERE/figures/" 2>/dev/null || true
mv -f *_results.json results_data.txt sensitivity_rows.tex ablation_rows.tex \
      "$HERE/results/" 2>/dev/null || true

echo "==== reproduction finished $(date) ===="
echo "Figures  -> figures/   Numerical outputs -> results/"
