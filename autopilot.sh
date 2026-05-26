#!/bin/bash
# Self-driving orchestrator: wait for downloads, set up env, run 4 experiments, aggregate.
# Designed to be launched via nohup/setsid so it survives ssh disconnects.
set -uo pipefail
LOG=/home/novarch2/workspace/sdb_sprint/autopilot.log
exec >> "$LOG" 2>&1

ts() { date '+%Y-%m-%d %H:%M:%S'; }
say() { echo "[$(ts)] $*"; }

HOME_DIR=/home/novarch2/workspace
ZIP="$HOME_DIR/data/MagicBathyNet.zip"
ENV_DIR="$HOME_DIR/miniforge3/envs/sdb"
EXPECTED_MD5="f7d5bf1873b846a96de00c722b7a1fd5"
EXPECTED_SIZE=5945227754

say "===== autopilot start ====="

# -------- 1. wait for aria2c to finish & verify zip --------
say "phase 1: wait for dataset download"
while pgrep -f aria2c >/dev/null; do
  sleep 60
done
say "aria2c no longer running"

# Verify size/MD5; if mismatch try one more download attempt
verify_zip() {
  local sz md5
  sz=$(stat -c %s "$ZIP" 2>/dev/null || echo 0)
  md5=$(md5sum "$ZIP" 2>/dev/null | awk '{print $1}')
  say "zip size=$sz expected=$EXPECTED_SIZE md5=$md5 expected=$EXPECTED_MD5"
  [ "$sz" = "$EXPECTED_SIZE" ] && [ "$md5" = "$EXPECTED_MD5" ]
}

if ! verify_zip; then
  say "verification failed, restarting aria2c"
  cd "$HOME_DIR/data"
  for i in 1 2 3; do
    aria2c -x 16 -s 16 -j 1 --max-tries=50 --retry-wait=5 --console-log-level=warn \
      -c -o MagicBathyNet.zip \
      https://zenodo.org/api/records/16753753/files/MagicBathyNet.zip/content && break
    sleep 10
  done
  if ! verify_zip; then
    say "ERROR: dataset zip still invalid after retries; aborting"
    exit 1
  fi
fi
say "dataset OK"

# -------- 2. extract --------
say "phase 2: extract dataset"
cd "$HOME_DIR/data"
if [ ! -d magicbathynet ]; then
  mkdir -p magicbathynet
  # Try multiple extractors
  if unzip -q MagicBathyNet.zip -d magicbathynet/ 2>>"$LOG"; then
    say "unzip succeeded"
  elif 7z x -y -omagicbathynet MagicBathyNet.zip >>"$LOG" 2>&1; then
    say "7z succeeded"
  else
    say "ERROR: extraction failed"
    exit 1
  fi
  # If the zip contains a top-level folder, normalize
  inner=$(ls magicbathynet | head -1)
  if [ -d "magicbathynet/$inner" ] && [ ! -d "magicbathynet/agia_napa" ]; then
    say "moving inner $inner/* up"
    mv magicbathynet/"$inner"/* magicbathynet/ 2>/dev/null || true
    rmdir magicbathynet/"$inner" 2>/dev/null || true
  fi
fi
say "extracted; tree top:"
ls magicbathynet/ | head -20 | sed 's/^/  /'

# -------- 3. wait for conda env then top up missing packages --------
say "phase 3: wait for conda env"
while pgrep -f "conda create" >/dev/null; do
  sleep 60
done
if [ ! -d "$ENV_DIR" ]; then
  say "conda create finished but no env dir; restarting"
  "$HOME_DIR/miniforge3/bin/conda" create -n sdb -c conda-forge -y python=3.11 \
    pytorch-gpu torchvision rasterio numpy scipy scikit-learn matplotlib tqdm \
    pandas tifffile pyyaml pillow einops || { say "ERROR creating env"; exit 1; }
fi
say "ensure scikit-image present"
"$HOME_DIR/miniforge3/bin/conda" install -n sdb -c conda-forge -y scikit-image 2>>"$LOG"

# -------- 4. verify GPU --------
say "phase 4: GPU + torch sanity"
source "$HOME_DIR/miniforge3/etc/profile.d/conda.sh"
conda activate sdb
python -c "import torch; print('torch', torch.__version__, 'cuda?', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')" || { say "ERROR torch broken"; exit 1; }

# -------- 5. find dataset root --------
ROOT_GUESS="$HOME_DIR/data/magicbathynet"
if [ ! -d "$ROOT_GUESS/agia_napa" ]; then
  CANDIDATE=$(find "$HOME_DIR/data/magicbathynet" -maxdepth 4 -type d -name agia_napa 2>/dev/null | head -1)
  if [ -n "$CANDIDATE" ]; then
    ROOT_GUESS=$(dirname "$CANDIDATE")
  fi
fi
say "dataset root = $ROOT_GUESS"
ls "$ROOT_GUESS" | head -10 | sed 's/^/  /'

# -------- 6. run the 4 experiments + aggregate --------
say "phase 6: run experiments"
export DATA_ROOT="$ROOT_GUESS"
export SDB_HOME=/home/novarch2/workspace/sdb_sprint
export RUNS="$SDB_HOME/runs"
export EPOCHS=${EPOCHS:-30}
bash "$SDB_HOME/run_all.sh" || say "run_all.sh exited non-zero"

say "===== autopilot done ====="
