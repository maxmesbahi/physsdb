#!/bin/bash
# PhysSDB autopilot: train PhysSDB on both sites, run 3 cross-site evals, aggregate.
set -uo pipefail
LOG=/home/novarch2/workspace/sdb_sprint/phys_autopilot.log
exec >> "$LOG" 2>&1

ts() { date '+%Y-%m-%d %H:%M:%S'; }
say() { echo "[$(ts)] $*"; }

HOME_DIR=/home/novarch2/workspace
DATA_ROOT="$HOME_DIR/data/magicbathynet"
RUNS="$HOME_DIR/sdb_sprint/runs"
SDB_HOME="$HOME_DIR/sdb_sprint"
EPOCHS=${EPOCHS:-40}
mkdir -p "$RUNS"

say "===== PhysSDB autopilot start ====="

source "$HOME_DIR/miniforge3/etc/profile.d/conda.sh"
conda activate sdb
cd "$SDB_HOME"

say "phase 1: PhysSDB Agia Napa"
if [ ! -f "$RUNS/physsdb_an/metrics.json" ]; then
  python train_phys.py --root "$DATA_ROOT" --area agia_napa --epochs "$EPOCHS" \
    --lambda-phys 1.0 --out "$RUNS/physsdb_an" 2>&1 | tee "$RUNS/physsdb_an.log" \
    || say "physsdb_an FAILED"
else
  say "physsdb_an exists; skipping"
fi

say "phase 2: PhysSDB Puck Lagoon"
if [ ! -f "$RUNS/physsdb_pl/metrics.json" ]; then
  python train_phys.py --root "$DATA_ROOT" --area puck_lagoon --epochs "$EPOCHS" \
    --lambda-phys 1.0 --out "$RUNS/physsdb_pl" 2>&1 | tee "$RUNS/physsdb_pl.log" \
    || say "physsdb_pl FAILED"
else
  say "physsdb_pl exists; skipping"
fi

say "phase 3: cross-site (AN → PL) for baseline / Stumpf / PhysSDB"
# baseline (trained on AN, eval on PL)
if [ ! -f "$RUNS/cross_baseline_an2pl/metrics.json" ] && [ -f "$RUNS/baseline_an/model_best.pt" ]; then
  python cross_site.py --ckpt "$RUNS/baseline_an/model_best.pt" \
    --model unet --in-channels 3 --stumpf 0 \
    --src-area agia_napa --dst-area puck_lagoon \
    --out "$RUNS/cross_baseline_an2pl" 2>&1 | tee "$RUNS/cross_baseline_an2pl.log" \
    || say "cross_baseline_an2pl FAILED"
fi
# Stumpf
if [ ! -f "$RUNS/cross_stumpf_an2pl/metrics.json" ] && [ -f "$RUNS/stumpf_an/model_best.pt" ]; then
  python cross_site.py --ckpt "$RUNS/stumpf_an/model_best.pt" \
    --model unet --in-channels 5 --stumpf 1 \
    --src-area agia_napa --dst-area puck_lagoon \
    --out "$RUNS/cross_stumpf_an2pl" 2>&1 | tee "$RUNS/cross_stumpf_an2pl.log" \
    || say "cross_stumpf_an2pl FAILED"
fi
# PhysSDB
if [ ! -f "$RUNS/cross_physsdb_an2pl/metrics.json" ] && [ -f "$RUNS/physsdb_an/model_best.pt" ]; then
  python cross_site.py --ckpt "$RUNS/physsdb_an/model_best.pt" \
    --model phys --in-channels 3 \
    --src-area agia_napa --dst-area puck_lagoon \
    --out "$RUNS/cross_physsdb_an2pl" 2>&1 | tee "$RUNS/cross_physsdb_an2pl.log" \
    || say "cross_physsdb_an2pl FAILED"
fi

say "phase 4: aggregate v2"
python aggregate_v2.py --runs "$RUNS" --out "$RUNS/summary_v2.md" || say "aggregate v2 FAILED"

say "===== PhysSDB autopilot done ====="
echo
cat "$RUNS/summary_v2.md" || true
