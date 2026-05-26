#!/usr/bin/env bash
# =============================================================================
#  remote-deploy.sh — runs on the target VPS during a GitHub Actions deploy.
#
#  Three modes:
#     MODE=ghcr   (default)  Pull pre-built image from ghcr.io and run it.
#                            Fast (~2-5 min first deploy, ~5s thereafter).
#                            Requires that the publish-image workflow has run
#                            at least once and that the image is public, OR
#                            that GHCR_USER/GHCR_TOKEN are provided for login.
#     MODE=build              Build the image on the VPS using the rsync'd
#                            source tree. Slow (~10-15 min first build). Works
#                            on VPSs that can't reach ghcr.io but can reach
#                            Docker Hub for the pytorch/pytorch base image.
#     MODE=conda              No Docker at all. Installs miniforge + a
#                            conda-forge env in the user's home, runs the app
#                            as a systemd --user service. Works on networks
#                            that block both ghcr.io AND Docker Hub (e.g.
#                            some Iranian / restricted-region VPSs), because
#                            github.com + conda-forge are usually still open.
#
#  Environment expected (set by the calling ssh command):
#     PW             VPS user's password (used for sudo)
#     REMOTE_DIR     where on the VPS the repo lives (relative to $HOME)
#     PROFILE        "gpu" or "cpu"
#     PORT           host port to publish (default 7860)
#     MODE           "ghcr" or "build" (default ghcr)
#     GHCR_IMAGE     e.g. ghcr.io/<owner>/<repo>      (required in ghcr mode)
#     GHCR_TAG       e.g. gpu | cpu | gpu-abc1234     (default = $PROFILE)
#     GHCR_USER      username for GHCR login          (only if image is private)
#     GHCR_TOKEN     token  for GHCR login            (only if image is private)
# =============================================================================
set -euo pipefail

: "${PW:?PW env not set}"
: "${REMOTE_DIR:?REMOTE_DIR env not set}"
: "${PROFILE:?PROFILE env not set}"
: "${PORT:?PORT env not set}"
MODE="${MODE:-ghcr}"

S() { echo "$PW" | sudo -S -p '' "$@"; }
HAVE() { command -v "$1" >/dev/null 2>&1; }
log() { printf "\n=== %s ===\n" "$*"; }

cd "$HOME/$REMOTE_DIR"

# ---------- 0. report environment -------------------------------------------
log "host: $(hostname)  $(uname -srm)"
log "disk free at $HOME"
df -h "$HOME" | tail -1
log "memory"
free -h | head -2
log "GPU"
if HAVE nvidia-smi; then
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv | head -5
else
  echo "(no nvidia-smi found)"
fi
log "deploy parameters"
echo "  MODE=$MODE  PROFILE=$PROFILE  PORT=$PORT  REMOTE_DIR=$REMOTE_DIR"
[ "$MODE" = "ghcr" ] && echo "  GHCR_IMAGE=${GHCR_IMAGE:-}  GHCR_TAG=${GHCR_TAG:-}"

# =============================================================================
#  CONDA MODE — fully separate code path; returns at the end without touching
#  the Docker logic below.
# =============================================================================
if [ "$MODE" = "conda" ]; then
  PORT_REQ="$PORT"
  MINIFORGE_DIR="$HOME/$REMOTE_DIR/miniforge3"
  ENV_DIR="$MINIFORGE_DIR/envs/sdb"
  CONDA_BIN="$MINIFORGE_DIR/bin/conda"
  ENV_PY="$ENV_DIR/bin/python"
  APP_DIR="$HOME/$REMOTE_DIR"

  # 1. install miniforge if missing
  if [ ! -x "$CONDA_BIN" ]; then
    log "Downloading miniforge installer from GitHub Releases"
    cd "$HOME/$REMOTE_DIR"
    for i in 1 2 3 4; do
      curl -L --max-time 600 --retry 3 --retry-delay 5 -C - \
        -o Miniforge3-Linux-x86_64.sh \
        https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh \
        && break
      sleep 5
    done
    log "Installing miniforge to $MINIFORGE_DIR"
    bash Miniforge3-Linux-x86_64.sh -b -p "$MINIFORGE_DIR"
    rm -f Miniforge3-Linux-x86_64.sh
  else
    log "miniforge already present at $MINIFORGE_DIR"
  fi

  # 2. create env if missing
  if [ ! -x "$ENV_PY" ]; then
    log "Creating conda env 'sdb' from conda-forge (this takes 5-12 min)"
    if [ "$PROFILE" = "gpu" ]; then PYTORCH_PKG="pytorch-gpu"; else PYTORCH_PKG="pytorch"; fi
    "$CONDA_BIN" create -n sdb -c conda-forge -y python=3.11 \
        "$PYTORCH_PKG" torchvision rasterio numpy scipy scikit-image \
        scikit-learn pandas matplotlib pillow mgrs pyproj einops \
        tifffile tqdm pyyaml gradio
  else
    log "conda env 'sdb' already exists; ensuring deps are present"
    "$CONDA_BIN" install -n sdb -c conda-forge -y --quiet gradio mgrs pyproj \
        scikit-image rasterio einops >/dev/null 2>&1 || true
  fi

  # 3. port auto-discovery (same logic as Docker mode)
  port_in_use() { S ss -lntp 2>/dev/null | awk '{print $4}' | grep -qE "(^|[.:])$1\$"; }
  if port_in_use "$PORT"; then
    log "Requested port $PORT is occupied; scanning for next free port"
    found=""
    for p in $(seq "$PORT" $((PORT + 50))); do
      port_in_use "$p" || { found="$p"; break; }
    done
    [ -z "$found" ] && { echo "::error::no free port found"; exit 1; }
    PORT="$found"
    echo "::notice::requested port $PORT_REQ was busy → publishing on $PORT instead"
  fi
  echo "$PORT" > "$APP_DIR/.actual-port"
  if HAVE ufw && S ufw status 2>/dev/null | grep -q "Status: active"; then
    S ufw allow "$PORT/tcp" >/dev/null 2>&1 || true
  fi

  # 4. ensure linger so user-level systemd survives this ssh logout
  S loginctl enable-linger "$(whoami)" >/dev/null 2>&1 || true
  export XDG_RUNTIME_DIR=/run/user/$(id -u)

  # 5. write the systemd --user unit
  log "Writing systemd --user unit ~/.config/systemd/user/sdb-app.service"
  mkdir -p "$HOME/.config/systemd/user"
  cat > "$HOME/.config/systemd/user/sdb-app.service" <<UNIT
[Unit]
Description=PhysSDB Sentinel-2 bathymetry dashboard (conda mode)
After=default.target

[Service]
Type=simple
WorkingDirectory=$APP_DIR
ExecStart=$ENV_PY -u $APP_DIR/app.py --host 0.0.0.0 --port $PORT
Restart=on-failure
RestartSec=5
StandardOutput=append:$APP_DIR/app.log
StandardError=append:$APP_DIR/app.log
Environment=HOME=$HOME
Environment=SDB_MODEL_ROOT=$APP_DIR/runs
Environment=SDB_DATA_ROOT=$APP_DIR/data/sample
Environment=GRADIO_ANALYTICS_ENABLED=False

[Install]
WantedBy=default.target
UNIT

  systemctl --user daemon-reload
  systemctl --user enable sdb-app >/dev/null 2>&1 || true
  : > "$APP_DIR/app.log"
  systemctl --user restart sdb-app

  # 6. wait for health
  log "Waiting up to 2 min for the dashboard to respond"
  ok=0
  for i in $(seq 1 12); do
    if curl -fsS "http://localhost:$PORT/" >/dev/null 2>&1; then
      echo "Dashboard responding on http://localhost:$PORT/ (after $((i*10))s)"
      ok=1; break
    fi
    sleep 10
  done

  log "Service status"
  systemctl --user status sdb-app --no-pager 2>&1 | head -15
  log "Last 30 log lines"
  tail -30 "$APP_DIR/app.log" 2>/dev/null || true

  if [ "$ok" != "1" ]; then
    echo "::error::Dashboard did not become healthy within 2 min."
    exit 1
  fi
  echo "Deployment OK (conda mode, $PROFILE profile, port $PORT)."
  exit 0
fi

# =============================================================================
#  DOCKER MODES — ghcr | build
# =============================================================================

# ---------- 1. Docker -------------------------------------------------------
if ! HAVE docker; then
  log "Installing Docker"
  curl -fsSL https://get.docker.com | S sh
  S systemctl enable --now docker
fi
S usermod -aG docker "$(whoami)" >/dev/null 2>&1 || true
log "Docker version"
S docker --version
S docker compose version || { echo "::error::docker compose plugin missing"; exit 1; }

# ---------- 2. NVIDIA Container Toolkit (only for GPU profile) --------------
if [ "$PROFILE" = "gpu" ] && HAVE nvidia-smi; then
  if ! S docker info 2>/dev/null | grep -q "Runtimes:.*nvidia"; then
    log "Installing NVIDIA Container Toolkit"
    distribution=$(. /etc/os-release; echo "$ID$VERSION_ID")
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
      | S gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
    curl -s -L "https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list" \
      | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
      | S tee /etc/apt/sources.list.d/nvidia-container-toolkit.list >/dev/null
    S apt-get update -qq
    S apt-get install -y -qq nvidia-container-toolkit
    S nvidia-ctk runtime configure --runtime=docker
    S systemctl restart docker
  else
    log "NVIDIA runtime already registered with Docker"
  fi
elif [ "$PROFILE" = "gpu" ] && ! HAVE nvidia-smi; then
  echo "::warning::profile=gpu requested but no nvidia-smi on host."
  echo "Container will start but fail to acquire a GPU. Re-run with profile=cpu"
  echo "or install a CUDA driver first."
fi

# ---------- 3. (UFW opened later, once we know the actual port) -------------
# moved below — see "port auto-discovery"

# ---------- 4. choose compose file & maybe login to GHCR --------------------
case "$MODE" in
  ghcr)
    : "${GHCR_IMAGE:?GHCR_IMAGE env not set (e.g. ghcr.io/<owner>/<repo>)}"
    export GHCR_IMAGE
    export GHCR_TAG="${GHCR_TAG:-$PROFILE}"
    COMPOSE_FILE="docker-compose.ghcr.yml"
    if [ -n "${GHCR_TOKEN:-}" ] && [ -n "${GHCR_USER:-}" ]; then
      log "Logging into ghcr.io as $GHCR_USER (private image mode)"
      echo "$GHCR_TOKEN" | S docker login ghcr.io -u "$GHCR_USER" --password-stdin
    fi
    ;;
  build)
    COMPOSE_FILE="docker-compose.yml"
    ;;
  *)
    echo "::error::unknown MODE=$MODE (expected ghcr|build)"
    exit 1
    ;;
esac
log "Using compose file: $COMPOSE_FILE"

# ---------- 5. (re)deploy ---------------------------------------------------
log "Stopping any previous instance of THIS compose project"
S env GHCR_IMAGE="${GHCR_IMAGE:-}" GHCR_TAG="${GHCR_TAG:-}" \
  docker compose -f "$COMPOSE_FILE" --profile "$PROFILE" down --remove-orphans 2>/dev/null || true

# ---- port auto-discovery ----
# If the requested host port is already bound (by another tenant, another
# container, the user's own previous deployment, etc.), scan upward for a
# free one in the same range.
port_in_use() {
  S ss -lntp 2>/dev/null | awk '{print $4}' | grep -qE "(^|[.:])$1\$"
}
REQUESTED_PORT="$PORT"
if port_in_use "$PORT"; then
  log "Requested port $PORT is occupied; scanning for next free port"
  found=""
  for p in $(seq "$PORT" $((PORT + 50))); do
    if ! port_in_use "$p"; then
      found="$p"; break
    fi
  done
  if [ -z "$found" ]; then
    echo "::error::no free port found in range $PORT-$((PORT + 50))"
    exit 1
  fi
  PORT="$found"
  echo "::notice::requested port $REQUESTED_PORT was busy → publishing on $PORT instead"
fi
# Record the actual port so the workflow can echo it back into the summary
echo "$PORT" > "$HOME/$REMOTE_DIR/.actual-port"
log "Publishing on host port $PORT (container internal port is always 7860)"
# Open the actual port in UFW if active
if HAVE ufw && S ufw status 2>/dev/null | grep -q "Status: active"; then
  S ufw allow "$PORT/tcp" >/dev/null 2>&1 || true
fi

if [ "$MODE" = "ghcr" ]; then
  log "Pulling image $GHCR_IMAGE:$GHCR_TAG"
  S env GHCR_IMAGE="$GHCR_IMAGE" GHCR_TAG="$GHCR_TAG" \
    docker compose -f "$COMPOSE_FILE" --profile "$PROFILE" pull
fi

log "Starting container"
S env SDB_PORT="$PORT" GHCR_IMAGE="${GHCR_IMAGE:-}" GHCR_TAG="${GHCR_TAG:-}" \
  docker compose -f "$COMPOSE_FILE" --profile "$PROFILE" up -d \
    $( [ "$MODE" = "build" ] && echo "--build" )

# ---------- 6. health wait --------------------------------------------------
log "Waiting up to 2 min for the dashboard to become healthy"
ok=0
for i in $(seq 1 12); do
  if curl -fsS "http://localhost:$PORT/" >/dev/null 2>&1; then
    echo "Dashboard responding on http://localhost:$PORT/ (after $((i*10))s)"
    ok=1
    break
  fi
  sleep 10
done

log "Container status"
S env GHCR_IMAGE="${GHCR_IMAGE:-}" GHCR_TAG="${GHCR_TAG:-}" \
  docker compose -f "$COMPOSE_FILE" --profile "$PROFILE" ps
log "Last 25 log lines"
S env GHCR_IMAGE="${GHCR_IMAGE:-}" GHCR_TAG="${GHCR_TAG:-}" \
  docker compose -f "$COMPOSE_FILE" --profile "$PROFILE" logs --tail=25 || true

if [ "$ok" != "1" ]; then
  echo "::error::Dashboard did not become healthy within 2 minutes."
  exit 1
fi
echo "Deployment OK ($MODE mode, $PROFILE profile, port $PORT)."
