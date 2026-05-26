# =============================================================================
#  Sentinel-2 SDB Research Dashboard — PhysSDB
#
#  One Dockerfile, two flavours:
#    # GPU (CUDA 12.4 + cudnn9, ~5 GB) — default
#    docker build -t physsdb:gpu .
#
#    # CPU-only (python:3.11-slim + pip torch CPU wheels, ~2 GB)
#    docker build \
#      --build-arg BASE_IMAGE=python:3.11-slim \
#      --build-arg INSTALL_TORCH=true \
#      -t physsdb:cpu .
#
#  Why two bases: pytorch/pytorch no longer publishes a `-cpu` tag, so for the
#  CPU build we start from a tiny Python image and pip-install the CPU torch
#  wheels.  GPU build uses PyTorch's official CUDA runtime so torch is already
#  present.
# =============================================================================

ARG BASE_IMAGE=pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime

FROM ${BASE_IMAGE}

LABEL org.opencontainers.image.title="PhysSDB SDB Dashboard"
LABEL org.opencontainers.image.description="Physics-Constrained Sentinel-2 Satellite-Derived Bathymetry"
LABEL org.opencontainers.image.source="https://github.com/<user>/physsdb"
LABEL org.opencontainers.image.licenses="MIT"

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MPLBACKEND=Agg

# --- OS deps ------------------------------------------------------------------
# rasterio's binary wheel bundles GDAL → only need a tiny set of system libs.
# tini  → PID 1 for clean signal handling
# curl  → healthcheck + diagnostics
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        tini \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# --- (CPU base only) install torch CPU wheels --------------------------------
# Re-declare ARG inside the build stage so RUN can read it.
ARG INSTALL_TORCH=false
RUN if [ "$INSTALL_TORCH" = "true" ]; then \
      pip install --index-url https://download.pytorch.org/whl/cpu \
                  torch==2.4.0 torchvision==0.19.0; \
    fi

# --- Python deps (own layer for cache hits) -----------------------------------
COPY requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt

# --- Application code ---------------------------------------------------------
COPY app.py             /app/
COPY bathy_truth.py     /app/
COPY s2_acquire.py      /app/
COPY train_sdb.py       /app/
COPY train_phys.py      /app/
COPY cross_site.py      /app/
COPY aggregate.py       /app/
COPY aggregate_v2.py    /app/

# --- Bundled model weights + sample tiles (optional) --------------------------
# These are copied if present at build time. For thin images, mount them at
# runtime instead via:  -v $(pwd)/runs:/app/runs -v $(pwd)/data:/app/data
# (the COPY lines below tolerate empty dirs and a "stub" sentinel file)
COPY runs/ /app/runs/
COPY data/ /app/data/

# --- Runtime config -----------------------------------------------------------
ENV SDB_MODEL_ROOT=/app/runs \
    SDB_DATA_ROOT=/app/data/sample \
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=7860 \
    GRADIO_ANALYTICS_ENABLED=False

# --- Non-root user ------------------------------------------------------------
RUN useradd -m -u 1001 sdb \
    && mkdir -p /app/runs /app/data /tmp/sdb_dash \
    && chown -R sdb:sdb /app /tmp/sdb_dash
USER sdb

EXPOSE 7860

# Liveness probe: GET / should return 200 once Gradio is up
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -fsS http://localhost:7860/ >/dev/null || exit 1

# tini handles SIGTERM/SIGINT cleanly so Ctrl-C and `docker stop` don't hang
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["python", "-u", "app.py", "--host", "0.0.0.0", "--port", "7860"]
