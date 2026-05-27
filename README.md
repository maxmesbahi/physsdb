# 🌊 PhysSDB — Sentinel-2 Satellite-Derived Bathymetry Dashboard

An interactive dashboard for estimating water depth from Sentinel-2 imagery
using **two methods side-by-side**:

| Method                | What it is                                                                                                                                                                                |
| --------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Baseline U-Net**    | Plain supervised regression (MagicBathyNet 2024).                                                                                                                                          |
| **PhysSDB** *(ours)*  | Multi-head U-Net + heteroscedastic uncertainty + a differentiable Lee-1998 forward model. Emits per-pixel depth, σ, optical-water parameters, and a physics-reconstructed RGB that doubles as a self-diagnostic. |

The dashboard lets users:

1. **Upload** a Sentinel-2 chip, or **draw an AOI on a map** and let the app find the lowest-cloud Sentinel-2 L2A scene from the past year.
2. Run both models on the chip → see depth maps + uncertainty + reconstruction side-by-side.
3. **Compare with measured bathymetry** (auto-fetched from NOAA Global DEM Mosaic for the same bbox).
4. Click any pixel to see the full predicted parameter vector + measured-vs-predicted error.

---

## ⚡ The 60-second deploy (TL;DR)

You only need a VPS, an SSH password for it, and a GitHub fork of this repo.

```text
1.  Fork  →  github.com/maxmesbahi/physsdb  →  "Fork"
2.  Repo Settings → Secrets and variables → Actions, add three:
       VPS_HOST       198.51.100.42
       VPS_USER       ubuntu               (or root, or whatever)
       VPS_PASSWORD   <your VPS password>
3.  Actions → "Deploy to VPS" → "Run workflow" → defaults are fine → click ▶
```

GitHub does everything for you: SSH into the VPS, install Docker if needed,
install the NVIDIA Container Toolkit if a GPU is present, pull the pre-built
image from `ghcr.io/maxmesbahi/physsdb:gpu`, auto-discover a free port,
start the container, run a healthcheck, and post the dashboard URL to the
workflow Summary.

**Time on a normal VPS**: 3–5 min (GHCR pull) · 10–15 min (local build) · 10–20 min (no-Docker conda fallback).

---

## 📦 What's in this repo

```
physsdb/
├── app.py                  Gradio dashboard (Upload + Map-acquire tabs)
├── train_sdb.py            Baseline U-Net training
├── train_phys.py           PhysSDB training (multi-head + physics loss)
├── cross_site.py           Cross-site evaluation
├── s2_acquire.py           Anonymous Sentinel-2 L2A acquisition (sentinel-cogs S3)
├── bathy_truth.py          Reference bathymetry from NOAA NGDC DEM mosaic
├── aggregate_v2.py         Aggregate per-run metrics into a summary
│
├── Dockerfile              Single Dockerfile (build-arg picks GPU vs CPU base)
├── docker-compose.yml      For local development: --profile gpu | --profile cpu
├── docker-compose.ghcr.yml For VPS deploy: pulls pre-built image from GHCR
├── requirements.txt        Python deps (PyTorch comes from the base image)
│
├── .github/workflows/
│   ├── publish-image.yml   Build + push images to GHCR (on every push to main)
│   ├── deploy.yml          SSH to a VPS and deploy
│   └── scripts/
│       └── remote-deploy.sh   Idempotent installer that runs on the VPS
│
├── runs/                   Pre-trained PhysSDB + baseline checkpoints (Git LFS)
├── data/sample/            Two MagicBathyNet sample tiles for the demo (Git LFS)
└── bin/                    Helper scripts (download-weights.sh, run-dev.sh)
```

---

## 🚀 Deployment paths — pick one

### Path A — GitHub Actions to a VPS (recommended for most users)

This is what the *60-second deploy* above describes. You don't touch the VPS
directly — GitHub runs the install for you.

#### Step-by-step

1. **Fork** this repo to your GitHub account.
2. **Set three repository secrets** at
   `Settings → Secrets and variables → Actions`:

   | Name         | Example                       | What it is                                                       |
   | ------------ | ----------------------------- | ---------------------------------------------------------------- |
   | `VPS_HOST`   | `198.51.100.42`               | Public IP or DNS of the target VPS                                |
   | `VPS_USER`   | `ubuntu`                      | Linux user with sudo (root, ubuntu, debian, ec2-user, …)          |
   | `VPS_PASSWORD` | `<plaintext>`               | Password for that user                                            |

   *Optional 4th secret:* `VPS_PORT` if SSH isn't on 22.

3. **Trigger the workflow**: `Actions → Deploy to VPS → Run workflow`.
   The defaults (`mode=ghcr`, `profile=gpu`, `port=7860`, `remote_dir=physsdb`) work for any VPS that can reach `ghcr.io`.

4. **Open the dashboard** at the URL shown in the workflow Summary
   (see [Accessing the dashboard](#-accessing-the-dashboard) for what to do
   if your VPS is behind NAT).

#### Three deploy modes

The workflow exposes a `mode` input that picks how the dashboard is installed on the VPS. **They produce the identical dashboard**; only the install path differs.

| `mode` | What it does | First deploy time | When to use |
| --- | --- | --- | --- |
| **`ghcr`** *(default)* | `docker pull ghcr.io/maxmesbahi/physsdb:<gpu\|cpu>` and run it. | **3–5 min** | Most VPSs (US/EU clouds). Fastest. |
| **`build`** | rsync source tree, `docker compose --build` on the VPS. PyTorch base pulled from Docker Hub. | 10–15 min | VPSs where `ghcr.io` is blocked but Docker Hub works. |
| **`conda`** | No Docker at all. Installs miniforge3 + conda-forge env, runs the app as a `systemd --user` service. | 10–20 min | VPSs where Docker registries are blocked. Also the simplest path for a non-Docker host. |

#### What the workflow does on the VPS

In order, idempotently — re-runs are safe:

1. SSH probe with `sshpass`.
2. Ship the appropriate files (2 files in `ghcr` mode, full repo in `build`/`conda`).
3. Install Docker if missing (`get.docker.com`). Skipped if already installed.
4. (`gpu` profile) install NVIDIA Container Toolkit if missing.
5. **Auto-discover a free host port** starting from your requested one (`7860 → 7861 → 7862 …` up to +50).
6. Open the chosen port in UFW if UFW is active.
7. Pull / build / install, start the container or `systemd --user` service.
8. Wait up to 2 min for the dashboard to return HTTP 200.
9. Write the actual port to `~/$REMOTE_DIR/.actual-port` on the VPS.
10. Post the clickable URL (or an SSH-tunnel command if the port isn't reachable from the public internet) to the workflow Summary.

#### Auto-deploy on push to `main`

Once a Build & Publish workflow run completes for `main`, the deploy workflow is triggered automatically (`workflow_run` trigger). You can also disable that behaviour by removing the `workflow_run:` block.

### Path B — Local Docker on a single machine

For a personal install on your own server or workstation:

```bash
# Prerequisites: Docker. Add NVIDIA Container Toolkit if you want GPU.

git lfs install
git clone https://github.com/maxmesbahi/physsdb.git
cd physsdb

# Option B1 — pull the pre-built image (fastest)
GHCR_IMAGE=ghcr.io/maxmesbahi/physsdb \
  docker compose -f docker-compose.ghcr.yml --profile gpu up -d

# Option B2 — build locally
docker compose --profile gpu up -d --build   # or: --profile cpu

# Open  http://localhost:7860/
```

### Path C — Local development (no Docker)

For iterating on the code:

```bash
git lfs install
git clone https://github.com/maxmesbahi/physsdb.git
cd physsdb

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install torch torchvision   # add --index-url https://download.pytorch.org/whl/cpu for CPU

./bin/run-dev.sh                # binds 0.0.0.0:7860
# Open  http://localhost:7860/
```

---

## 🌐 Accessing the dashboard

> **About the port**
> The deploy script **auto-discovers a free port** starting from the one you requested (default `7860`). If the requested port is busy, it scans upward and picks the first free one. **You never need to know the exact port in advance — three sources tell you:**
>
> 1. **Workflow Summary** at the bottom of the *Deploy to VPS* run page (most convenient). If auto-shift happened, the summary says so explicitly:
>    > Requested port: `7860` (was occupied)
>    > Actual port: `7862` (auto-selected)
> 2. **Marker file on the VPS**:
>    ```bash
>    ssh <VPS_USER>@<VPS_HOST> 'cat ~/physsdb/.actual-port'
>    ```
> 3. **`ss` on the VPS** (forensic fallback):
>    ```bash
>    ssh <VPS_USER>@<VPS_HOST> 'ss -lntp 2>/dev/null | grep python'
>    ```
>
> Substitute the **actual port** wherever you see `$PORT` below.

#### Case A — VPS has a public IP and `$PORT` is reachable from the internet

Open the URL the workflow Summary printed:

```
http://<VPS_HOST>:$PORT/
```

#### Case B — VPS is behind NAT, or cloud-provider firewall blocks `$PORT`

This is common on small VPSs and home servers. The workflow Summary instead prints an SSH-tunnel command:

```bash
# On your laptop:
ssh -L $PORT:127.0.0.1:$PORT $VPS_USER@$VPS_HOST
# (leave that session open)

# In your browser:
http://127.0.0.1:$PORT/
```

The Gradio app inside the container *always* binds the same internal port (7860); Docker / systemd is what maps it to whatever host `$PORT` was selected.

---

## 🔧 Configuration

All settings are env vars; defaults work out-of-the-box.

| Variable                   | Default              | What it does                                                 |
| -------------------------- | -------------------- | ------------------------------------------------------------ |
| `SDB_MODEL_ROOT`           | `/app/runs`          | Where the dashboard looks for checkpoints                     |
| `SDB_DATA_ROOT`            | `/app/data/sample`   | Where the dashboard looks for the sample tiles                |
| `GRADIO_SERVER_NAME`       | `0.0.0.0`            | Bind interface                                                |
| `GRADIO_SERVER_PORT`       | `7860`               | Bind port *inside* the container — host port is mapped separately |
| `GRADIO_TEMP_DIR`          | per-mode path        | Per-user gradio cache; avoids `/tmp/gradio` collisions on multi-tenant VPSs |
| `GRADIO_ANALYTICS_ENABLED` | `False`              | Disable Gradio's HuggingFace telemetry                        |
| `SDB_PORT`                 | `7860`               | (compose only) host port the container publishes              |

For GitHub Actions, the workflow inputs map to these via the deploy script — you don't normally set them directly.

---

## 🛠 Hardware

| Tier                                | Inference per chip (256×256) | Notes                              |
| ----------------------------------- | ---------------------------- | ---------------------------------- |
| GPU (RTX 3060 / T4 / V100+ / A100)  | < 0.5 s                      | Recommended for live demos          |
| CPU only (4 vCPU, 4 GB RAM)         | 1–3 s                        | Fine for a free-tier HF Space       |
| Disk                                | ~5 GB GPU image / ~2 GB CPU image / ~12 GB conda env | + 60 MB for weights                 |
| RAM at runtime                      | ~1.5 GB                      | Both models loaded                  |
| GPU memory                          | < 2 GB                       | A 4 GB GPU is plenty                |

---

## 🌐 Network — what the dashboard talks to

| Endpoint                                                     | When it's used                                  | If blocked                                         |
| ------------------------------------------------------------ | ----------------------------------------------- | -------------------------------------------------- |
| `sentinel-cogs.s3.us-west-2.amazonaws.com`                   | "Acquire from S3" tab — anonymous COG reads     | Acquisition tab is disabled; Upload tab still works |
| `gis.ngdc.noaa.gov`                                          | Bathymetry-truth comparison                     | Truth panels render as "no reference"               |
| `unpkg.com`                                                  | Leaflet + leaflet-draw on the map               | Map doesn't appear; coords can be typed manually    |
| `a.tile.openstreetmap.fr`                                    | Map background tiles                            | Map looks blank but draw still works                |

All requests are anonymous GETs. The dashboard never sends user data outbound.

---

## 🩺 Troubleshooting

| Symptom                                                                                    | Likely cause / fix                                                                                                                |
| ------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------- |
| Deploy fails on `Sanity-check required secrets`                                            | One of `VPS_HOST` / `VPS_USER` / `VPS_PASSWORD` is missing.                                                                       |
| `docker pull` from `ghcr.io` fails with **TCP reset** repeatedly                             | Your VPS network filters ghcr.io. Re-run with `mode=build` or `mode=conda`.                                                       |
| `docker pull` hits **502 / 403 Forbidden** on PyTorch base from a registry mirror              | The mirror is rate-limiting or blocking. Re-run with `mode=conda` (no Docker required).                                            |
| Conda env install hangs at `Transaction starting` for >30 min                               | Already handled — the script uses `mamba` (parallel solver) and a 2-attempt retry that cleans `.partial` files between attempts.   |
| Dashboard error: `Permission denied: '/tmp/gradio/...'`                                      | Another tenant on the same VPS owned `/tmp/gradio`. Already handled — every mode sets `GRADIO_TEMP_DIR` to a user-private path.   |
| Workflow finishes but `http://<VPS>:<port>/` doesn't open                                  | VPS is behind NAT or cloud firewall is closed. Use the SSH-tunnel command from the workflow Summary.                              |
| Map tab is blank                                                                            | Browser console (F12) shows whether `unpkg.com` or `openstreetmap.fr` failed. Hard-refresh with Ctrl-Shift-R.                      |
| `requested port 7860 was busy → publishing on 7861 instead` in the log                       | Working as intended — another service had 7860. Open the actual port from the Summary or `.actual-port`.                          |

---

## ⏯ Managing the dashboard after deploy

### Docker modes

```bash
ssh <VPS_USER>@<VPS_HOST>
cd ~/physsdb

docker compose --profile gpu ps                # status
docker compose --profile gpu logs --tail=50    # logs
docker compose --profile gpu down              # stop
docker compose --profile gpu up -d             # start
```

### Conda mode

```bash
ssh <VPS_USER>@<VPS_HOST>

# uid of the deploy user (1008 in the reference setup)
UID_=$(id -u)

XDG_RUNTIME_DIR=/run/user/$UID_ systemctl --user status  sdb-app
XDG_RUNTIME_DIR=/run/user/$UID_ systemctl --user restart sdb-app
XDG_RUNTIME_DIR=/run/user/$UID_ systemctl --user stop    sdb-app

tail -f ~/physsdb/app.log
```

### Update the deployment

Just `git push` to `main`. The workflow detects code changes, rebuilds the image on GHCR if needed, and re-runs the deploy. Container restarts; users get a Gradio reconnect popup but don't lose session state in the page.

### Roll back

`git revert` the bad commit, push, the workflow redeploys the previous version.

---

## 🎓 Reproducing the training

The pre-trained weights in `runs/` were produced by:

```bash
# 1. Download MagicBathyNet (~6 GB)
mkdir -p data && cd data
curl -L -o MagicBathyNet.zip \
  https://zenodo.org/api/records/16753753/files/MagicBathyNet.zip/content
unzip MagicBathyNet.zip -d magicbathynet/
cd ..

# 2. Train baseline + Stumpf + PhysSDB on both sites
bash run_all.sh
bash phys_autopilot.sh

# 3. Aggregate
python aggregate_v2.py --runs runs --out runs/summary_v2.md
```

~30 min total on an RTX 3090 (small dataset: 28 train + 7 test tiles per site at 18×18 px).

---

## 🔒 GHCR images — making them private (optional)

GHCR images are public by default after the first `Build & Publish` run. To
keep them private:

1. github.com → your profile → **Packages** → click `physsdb` → **Package settings** → **Change visibility** → **Private**.
2. Add two more secrets to your fork:

   | Name         | Value                                                                                                       |
   | ------------ | ----------------------------------------------------------------------------------------------------------- |
   | `GHCR_USER`  | Your GitHub username (lowercase)                                                                            |
   | `GHCR_TOKEN` | A Personal Access Token (classic) with scope **`read:packages`**                                            |

The deploy workflow auto-detects these and does `docker login ghcr.io` before pulling.

---

## 📚 Citing

If you use PhysSDB in academic work please cite the thesis / paper (TBD)
*and* the underlying dataset:

```bibtex
@inproceedings{agrafiotis2024magicbathynet,
  author    = {Agrafiotis, Panagiotis and Janowski, Łukasz and Skarlatos, Dimitrios and Demir, Begüm},
  title     = {MagicBathyNet: A Multimodal Remote Sensing Dataset for Bathymetry Prediction and Pixel-Based Classification in Shallow Waters},
  booktitle = {IGARSS 2024},
  pages     = {249--253},
  year      = {2024},
  doi       = {10.1109/IGARSS53475.2024.10641355}
}
```

## License

Code: **MIT** (see [LICENSE](LICENSE)).
Sample tiles: CC-BY-NC-SA 4.0 (MagicBathyNet original license).
Pre-trained weights: MIT — please cite if reused.
