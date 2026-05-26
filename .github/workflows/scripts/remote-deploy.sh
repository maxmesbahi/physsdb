#!/usr/bin/env bash
# =============================================================================
#  remote-deploy.sh — runs on the target VPS during a GitHub Actions deploy.
#
#  Two modes:
#     MODE=ghcr   (default)  Pull pre-built image from ghcr.io and run it.
#                            Fast (~2-5 min first deploy, ~5s thereafter).
#                            Requires that the publish-image workflow has run
#                            at least once and that the image is public, OR
#                            that GHCR_USER/GHCR_TOKEN are provided for login.
#     MODE=build              Build the image on the VPS using the rsync'd
#                            source tree (the original behaviour). Slow but
#                            works on VPSs that can't reach ghcr.io.
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

# ---------- 3. firewall (UFW only) ------------------------------------------
if HAVE ufw && S ufw status 2>/dev/null | grep -q "Status: active"; then
  log "Opening UFW port $PORT/tcp"
  S ufw allow "$PORT/tcp" || true
fi

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
log "Stopping any previous instance"
S env GHCR_IMAGE="${GHCR_IMAGE:-}" GHCR_TAG="${GHCR_TAG:-}" \
  docker compose -f "$COMPOSE_FILE" --profile "$PROFILE" down --remove-orphans 2>/dev/null || true

# Free the port if something else is on it
if S ss -lntp 2>/dev/null | awk '{print $4}' | grep -q ":$PORT$"; then
  echo "::warning::port $PORT is occupied by another process; container may fail to bind."
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
