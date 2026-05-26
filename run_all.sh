#!/bin/bash
# Runs the 4 experiments (baseline + Stumpf, on Agia Napa + Puck Lagoon)
# and aggregates the results table.
set -euo pipefail

SDB_HOME="${SDB_HOME:-/home/novarch2/workspace/sdb_sprint}"
DATA_ROOT="${DATA_ROOT:-/home/novarch2/workspace/data/magicbathynet}"
RUNS="${RUNS:-$SDB_HOME/runs}"
EPOCHS="${EPOCHS:-30}"
mkdir -p "$RUNS"

# Use the sdb conda env
source /home/novarch2/workspace/miniforge3/etc/profile.d/conda.sh
conda activate sdb

cd "$SDB_HOME"

run_one() {
  local area="$1" stumpf="$2" tag="$3"
  local out="$RUNS/$tag"
  if [ -f "$out/metrics.json" ]; then
    echo "=== skip $tag (already done) ==="
    return
  fi
  echo "=== $tag : area=$area stumpf=$stumpf ==="
  python train_sdb.py \
    --root "$DATA_ROOT" \
    --area "$area" \
    --modality s2 \
    --stumpf "$stumpf" \
    --epochs "$EPOCHS" \
    --batch-size 32 \
    --out "$out" 2>&1 | tee "$out.log" || echo "$tag FAILED"
}

run_one agia_napa   0 baseline_an
run_one agia_napa   1 stumpf_an
run_one puck_lagoon 0 baseline_pl
run_one puck_lagoon 1 stumpf_pl

python aggregate.py --runs "$RUNS" --out "$RUNS/summary.md"
echo "=== ALL DONE ==="
cat "$RUNS/summary.md"
