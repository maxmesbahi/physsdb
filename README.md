# 🌊 PhysSDB — Physics-Constrained Sentinel-2 Satellite-Derived Bathymetry

An end-to-end research dashboard for **estimating water depth from Sentinel-2
imagery** and comparing two methods side-by-side:

| Method                | What it is                                                        |
| --------------------- | ----------------------------------------------------------------- |
| **Baseline U-Net**    | Plain supervised regression (MagicBathyNet 2024 baseline).         |
| **PhysSDB** *(ours)*  | Multi-head U-Net + heteroscedastic NLL + differentiable Lee-1998 forward model. Produces depth, per-pixel σ, optical-water parameters, and a physics-reconstructed image used as a self-diagnostic. |

The interactive dashboard lets users:
1. **Upload** a Sentinel-2 chip, *or* **draw a rectangle on a map** and let the app fetch the lowest-cloud Sentinel-2 L2A scene for that area in the past year (anonymous S3 LIST → COG window read).
2. Run both models on the chip; see depths + uncertainty + reconstruction side-by-side.
3. **Compare against measured bathymetry** automatically fetched from NOAA's Global DEM Mosaic for the same bbox.
4. Click any pixel for the full predicted parameter vector + measured-vs-predicted error.

---

## TL;DR — Quick start

You only need **Docker** (and **NVIDIA Container Toolkit** if you want GPU):

```bash
# 1. Clone (with LFS so the weights come along)
git lfs install                                       # one-time per machine
git clone https://github.com/<USER>/physsdb.git
cd physsdb
git lfs pull                                          # if `runs/` looks empty

# 2a. GPU host
docker compose --profile gpu up -d

# 2b. CPU-only host (laptop / cheap VPS / HF Spaces target)
docker compose --profile cpu up -d

# 3. Open the dashboard
xdg-open http://localhost:7860     # or http://<server-ip>:7860
```

Cold start: ~30 s on GPU, ~60 s on CPU (model load + first Gradio request).

---

## What's in this repo

```
physsdb/
├── app.py                  Gradio dashboard (Upload + Acquisition tabs)
├── train_sdb.py            Baseline U-Net training
├── train_phys.py           PhysSDB training (physics-consistent multi-head)
├── cross_site.py           Cross-site evaluation (train AN → test PL, etc.)
├── s2_acquire.py           Anonymous Sentinel-2 L2A acquisition from sentinel-cogs S3
├── bathy_truth.py          Reference bathymetry from NOAA NGDC DEM Global Mosaic
├── aggregate_v2.py         Build the summary table comparing all runs
├── Dockerfile              Single Dockerfile (build-arg picks GPU vs CPU base)
├── docker-compose.yml      Two profiles: --profile gpu  /  --profile cpu
├── requirements.txt        Python deps (PyTorch is in the base image)
├── bin/
│   ├── download-weights.sh   Pull weights from a GitHub Release tarball
│   └── run-dev.sh            Run app.py locally without Docker
├── runs/                   Pre-trained PhysSDB + baseline checkpoints (Git LFS)
└── data/sample/            Two MagicBathyNet sample tiles for the demo (Git LFS)
```

---

## Architecture

```
┌────────────────────┐                                          ┌──────────────┐
│   User browser     │  HTTPS/HTTP (Gradio UI on :7860)         │  Leaflet     │
│   localhost:7860   │ ◄─────────────────────────────────────► │  + draw      │
└────────┬───────────┘                                          └──────┬───────┘
         │                                                              │ tile + draw events
         ▼                                                              ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  Docker container (physsdb:gpu | physsdb:cpu)                                │
│                                                                              │
│  app.py  ── Upload   ─►  preprocess + UNet_bathy + PhysSDB ─► panels         │
│         ── Acquire   ─►  s2_acquire ─► sentinel-cogs S3 (COG range reads)    │
│                          bathy_truth ─► NOAA NGDC DEM Global Mosaic          │
│                          UNet_bathy + PhysSDB ─► panels                       │
└──────────────────────────────────────────────────────────────────────────────┘
                                  ▲  ▲
                                  │  │
                ┌─────────────────┘  └──────────────────────┐
                │                                            │
   public S3 (anonymous LIST + COG range reads)    NOAA NGDC ArcGIS REST
   sentinel-cogs.s3.us-west-2.amazonaws.com        gis.ngdc.noaa.gov/.../exportImage
```

---

## Hardware requirements

| Tier                            | Inference latency (256×256) | Notes                                                     |
| ------------------------------- | --------------------------- | --------------------------------------------------------- |
| GPU (RTX 3060 / T4 / V100+ / A100) | < 0.5 s per chip            | Recommended for live demos.                                |
| CPU only (4 vCPU, 4 GB RAM)     | 1–3 s per chip              | Fine for an HF Space free tier or a $5 VPS.                |
| Disk (image + weights)          | ~5 GB (GPU) / ~2 GB (CPU)   | + ~60 MB for weights.                                      |
| RAM at runtime                  | ~1.5 GB                     | Tested footprint with both models loaded.                  |
| GPU memory                      | < 2 GB                      | A 4 GB GPU is plenty.                                      |

---

## Deployment scenarios (step-by-step)

### Scenario 1 — GPU VPS with a public IP (best for a live demo)

Tested on Ubuntu 22.04 + Docker 24 + NVIDIA Container Toolkit 1.14 + RTX 3090.

```bash
# 1. Install Docker
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER && newgrp docker

# 2. Install NVIDIA Container Toolkit (gives Docker access to the GPU)
distribution=$(. /etc/os-release; echo $ID$VERSION_ID)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt update && sudo apt install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# 3. Verify GPU is visible to Docker
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi

# 4. Clone + run
git lfs install
git clone https://github.com/<USER>/physsdb.git && cd physsdb
git lfs pull
docker compose --profile gpu up -d --build

# 5. Open the firewall port (if cloud provider has one) — UFW example:
sudo ufw allow 7860/tcp

# 6. Visit  http://<vps-ip>:7860
```

### Scenario 2 — GPU VPS **behind NAT** (private host, no inbound port)

Useful when your cloud provider only NATs port 22 (SSH) to the VM. Run the dashboard on the private port and reach it through an SSH tunnel from your laptop.

```bash
# On the VPS — same as Scenario 1, but skip the firewall step.
docker compose --profile gpu up -d --build

# On your laptop — open a port-forward tunnel
ssh -L 7860:127.0.0.1:7860 <user>@<vps-public-ip>

# Then open  http://127.0.0.1:7860  in your browser
```

This is also how you should expose the dashboard to your thesis committee privately.

### Scenario 3 — CPU-only VPS (no GPU)

```bash
git lfs install
git clone https://github.com/<USER>/physsdb.git && cd physsdb
git lfs pull
docker compose --profile cpu up -d --build
```

The CPU image is ~2 GB vs ~5 GB for the GPU one. Inference is 1–3 s per chip — fast enough for a demo.

### Scenario 4 — Network-restricted VPS (e.g. some regions where PyPI / pytorch.org / google domains are blocked)

Two strategies:

**4a — Build the image elsewhere and `docker save` / `docker load` onto the VPS:**

```bash
# On a machine WITH unrestricted internet:
docker compose --profile gpu build
docker save physsdb:gpu | gzip > physsdb-gpu.tar.gz
scp physsdb-gpu.tar.gz user@restricted-vps:/tmp/

# On the restricted VPS:
docker load < /tmp/physsdb-gpu.tar.gz
docker compose --profile gpu up -d
```

**4b — Build on the VPS using only conda-forge** (the path I followed for the original RTX 3090 VPS that blocked pypi/pytorch.org). Replace pip+PyTorch base with miniforge + conda-forge:

See [`docs/build-on-restricted-vps.md`](docs/build-on-restricted-vps.md) for the exact recipe (todo — add if needed).

### Scenario 5 — Local laptop (dev / iteration)

Use a Python venv directly without Docker for the fastest edit-rerun cycle:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt torch torchvision   # add --extra-index-url for CPU/GPU as needed
./bin/run-dev.sh                                    # binds 0.0.0.0:7860
```

### Scenario 6 — Hugging Face Spaces (zero-infrastructure public demo)

This Docker setup is **not** directly compatible with HF Spaces (which uses its own build system) but porting is mechanical:

1. Create a Gradio Space on huggingface.co.
2. Copy `app.py`, `*.py`, `requirements.txt`, `runs/`, `data/sample/` into the Space repo.
3. Add an HF-compatible `README.md` frontmatter:

```yaml
---
title: PhysSDB Demo
sdk: gradio
sdk_version: 5.0.0
app_file: app.py
license: mit
---
```

4. `git push` to the Space remote. HF auto-builds and serves at `https://huggingface.co/spaces/<user>/physsdb`. Choose **CPU basic** (free) or **T4 small** ($0.60/h).

---

## Configuration (environment variables)

| Variable                    | Default                                       | Used for                                |
| --------------------------- | --------------------------------------------- | --------------------------------------- |
| `SDB_MODEL_ROOT`            | `/app/runs`                                   | Where the dashboard finds checkpoints.  |
| `SDB_DATA_ROOT`             | `/app/data`                                   | Where the sample tiles live.            |
| `GRADIO_SERVER_NAME`        | `0.0.0.0`                                     | Bind interface.                         |
| `GRADIO_SERVER_PORT`        | `7860`                                        | Bind port.                              |
| `GRADIO_ANALYTICS_ENABLED`  | `False`                                       | Disable HF Gradio telemetry.            |
| `SDB_PORT` (compose only)   | `7860`                                        | Host port published by `docker compose`. |

---

## Filesystem layout — `runs/` and `data/`

```
runs/                         ← Git LFS  (~60 MB total)
  baseline_an/model_best.pt
  baseline_pl/model_best.pt
  physsdb_an/model_best.pt
  physsdb_pl/model_best.pt

data/sample/                  ← Git LFS (~50 KB total)
  agia_napa/
    img/s2/img_411.tif
    depth/s2/depth_411.tif
    norm_param_s2_an.txt
  puck_lagoon/
    img/s2/img_411.tif
    depth/s2/depth_411.tif
    norm_param_s2_pl.txt
```

If you want to mount your own (e.g. with your own trained weights), use:

```bash
docker run --rm --gpus all -p 7860:7860 \
  -v /my/own/runs:/app/runs:ro \
  -v /my/own/data:/app/data:ro \
  physsdb:gpu
```

---

## Network behaviour

The dashboard makes outbound calls to:

| Host                                                        | Why                                | If blocked                                                |
| ----------------------------------------------------------- | ---------------------------------- | --------------------------------------------------------- |
| `sentinel-cogs.s3.us-west-2.amazonaws.com`                  | Acquire Sentinel-2 L2A chips        | Acquisition tab won't work; Upload tab still works.       |
| `gis.ngdc.noaa.gov`                                         | Fetch measured bathymetry truth     | Truth panels render as "no reference"; everything else works. |
| `unpkg.com`                                                 | Leaflet + leaflet-draw JS/CSS       | Map won't appear; coordinates can be typed manually.       |
| `a.tile.openstreetmap.fr`                                   | Background map tiles                | Map appears blank but draw still works.                    |

All requests are anonymous (no API keys), GET-only, and the dashboard never sends user data outbound.

---

## Reproducing the training

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

# 3. Aggregate results into a single summary
python aggregate_v2.py --runs runs --out runs/summary_v2.md
```

This takes ~30 min total on an RTX 3090 (training is GPU-bound but the dataset is tiny: 28 train + 7 test tiles per site at 18×18 px).

---

## Troubleshooting

| Symptom                                                | Fix |
|--------------------------------------------------------|-----|
| `docker: Error response from daemon: could not select device driver`  | NVIDIA Container Toolkit not installed (see Scenario 1, step 2). |
| Container exits immediately with `ImportError: ...`    | The base image probably mismatches your CUDA driver. Try the CPU profile, or use `pytorch/pytorch:2.4.0-cuda11.8-cudnn9-runtime` build-arg if your driver is older. |
| Map tab is blank                                       | Browser console (F12) → check for blocked `unpkg.com` / `openstreetmap.fr`. Hard-refresh with Ctrl-Shift-R. |
| "No scene matched the filter" on Acquisition          | Increase the date window slider; raise max cloud %; draw the rectangle entirely over water. |
| "Truth fetch failed"                                    | NOAA NGDC is rate-limited; retry in a few seconds. |
| "ConnectionRefusedError" hitting the host port         | The container bound the right port but a host firewall is blocking; `sudo ufw allow 7860/tcp` (UFW) or open the port in your cloud provider's panel. |
| Disk usage too high                                     | Run `docker system prune -a -f` to reclaim dangling images and old layers. |

---

## One-click deployment via GitHub Actions (CI/CD)

Two workflows ship together:

| Workflow                                                                | What it does                                                                                          |
| ----------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------- |
| [`.github/workflows/publish-image.yml`](.github/workflows/publish-image.yml) | Builds `physsdb:gpu` and `physsdb:cpu` images on GitHub-hosted runners and pushes them to **GitHub Container Registry** (ghcr.io/&lt;owner&gt;/&lt;repo&gt;). Triggered on every push to `main` and on every Release tag. |
| [`.github/workflows/deploy.yml`](.github/workflows/deploy.yml)             | SSHs into a VPS and runs **one of three deploy modes** (see below). Triggered manually or after a fresh publish.                                          |

### The three deploy modes

| `mode` | What it does | First-deploy time | When to use it |
| --- | --- | --- | --- |
| `ghcr` *(default)* | `docker pull` from `ghcr.io/<owner>/<repo>:<gpu\|cpu>`; `docker compose up -d`. Image already has weights baked in. | ~3–5 min | Any VPS that can reach **ghcr.io** (most US/EU clouds). Fastest. |
| `build` | rsync the source tree → `docker compose --build` on the VPS, pulling the PyTorch base from **Docker Hub**. | ~10–15 min | VPSs where **ghcr.io is blocked** but Docker Hub works. |
| `conda` | No Docker at all. Installs miniforge in `~/$REMOTE_DIR/miniforge3`, creates a conda-forge env, runs the app as a `systemd --user` service. | ~10–15 min | VPSs where **both** ghcr.io **and** Docker Hub are blocked / filtered (e.g. some Iranian or restricted-region VPSs). Works on **any** host with internet to GitHub Releases + conda-forge — also works on normal VPSs as a Docker-free option. |

Pipeline:

```
   ┌────────────────┐    push to main    ┌────────────────────┐
   │   GitHub repo  │ ──────────────────►│ publish-image.yml  │  (build & push)
   └────────────────┘                    │   builds 2 images  │
                                          │   pushes to GHCR   │
                                          └────────┬───────────┘
                                                   │ workflow_run
                                                   ▼
                                          ┌────────────────────┐
                                          │    deploy.yml      │
                                          │  ssh → VPS         │
                                          │  docker compose    │
                                          │     pull && up -d  │
                                          └────────┬───────────┘
                                                   ▼
                                          ┌────────────────────┐
                                          │  Dashboard live    │
                                          │  on http://VPS:7860│
                                          └────────────────────┘
```

### ⚠️ Security trade-off — read this first

Storing a VPS password as a GitHub secret is convenient but riskier than SSH-key auth:

- Anyone with **write access to the repo or its Actions secrets** can read the password (GitHub admins can also view it indirectly via masked-log timing attacks; treat secrets as exposed to anyone you give push access).
- GitHub-hosted runners have outbound internet but you should still scope the user to **one VPS** — never reuse the password elsewhere.
- After this workflow lands, **switch to SSH key auth** (`ssh-copy-id` from a local machine + change `PreferredAuthentications=publickey` in `/etc/ssh/sshd_config`) and replace the `VPS_PASSWORD` secret with an `SSH_PRIVATE_KEY` one. You only need passwords for the very first deploy.

### Setup (one time, ~3 min)

1. In your repo on github.com, go to **Settings → Secrets and variables → Actions**.
2. Add three Repository secrets:

   | Name           | Value                                    |
   |----------------|------------------------------------------|
   | `VPS_HOST`     | Public IP or DNS of the VPS, e.g. `198.51.100.42` |
   | `VPS_USER`     | A Linux account with sudo, e.g. `root`, `ubuntu`, `debian` |
   | `VPS_PASSWORD` | The password for that account             |

   Optional:

   | Name        | Value                          |
   |-------------|--------------------------------|
   | `VPS_PORT`  | Non-standard SSH port (default `22`) |

3. *(optional)* In **Settings → Environments**, create an Environment called `production` and move the same secrets in there. This lets you require manual approval before each deploy. (If you skip this, comment the `environment: production` line out of `deploy.yml`.)

### Run the deploy

The workflow has two triggers:

| Trigger                        | When it runs                                                         |
|--------------------------------|----------------------------------------------------------------------|
| **Manual** (`workflow_dispatch`) | Actions tab → **Deploy to VPS** → **Run workflow** → pick profile/port |
| **On push to `main`**           | Auto-deploys after any code change (skips doc-only changes)          |

What it does, in order:

1. Checkout the repo (with LFS so model weights come along).
2. SSH into the VPS with `sshpass` (password auth, host key auto-accepted on first run).
3. `rsync` the repo to `~/$REMOTE_DIR` on the VPS.
4. Install **Docker** if missing (official `get.docker.com` script).
5. For the **GPU profile**, install the **NVIDIA Container Toolkit** if missing.
6. Open port `7860/tcp` in **UFW** if UFW is active.
7. `docker compose --profile <gpu|cpu> up -d --build`.
8. Wait up to 2 min for `http://localhost:7860/` to return 200.
9. Try to reach `http://<VPS_HOST>:<port>/` from the GitHub-hosted runner and post a clickable URL — or a fallback SSH-tunnel command — to the workflow Summary.

### What you'll see in the Actions summary

On a happy public-IP deploy:

> ## PhysSDB deployment result
> **Profile**: `gpu` &nbsp; **Remote dir**: `~/physsdb` &nbsp; **Port**: `7860`
>
> ### ✅ Dashboard reachable
> Open in your browser: **http://198.51.100.42:7860/**

On a NAT'd VPS (only port 22 forwarded):

> ### ⚠️ Container is up, but port 7860 is **not** reachable from the public internet
>
> Access via SSH tunnel:
> ```bash
> ssh -L 7860:127.0.0.1:7860 ubuntu@198.51.100.42
> # then open http://127.0.0.1:7860/
> ```

### First-run order (what the user does, end-to-end)

1. **Fork or push this repo** to GitHub (see push recipe at the bottom of the file).
2. **Push to `main`** → `publish-image.yml` runs, builds two images, pushes them to
   `ghcr.io/<owner>/<repo>:gpu` and `:cpu`. (~15 min first time, ~3 min thereafter.)
3. **Make the images public** *(highly recommended)*:
   - github.com → your profile → **Packages** → click the `physsdb` package → **Package settings** →
     "Change visibility" → **Public**.
   - With public images, the deploy workflow does not need a GHCR token (anonymous pull).
4. **Add the three VPS secrets** (above).
5. **Actions tab → "Deploy to VPS" → Run workflow** → defaults to `mode=ghcr`, `profile=gpu`, `port=7860`. Click **Run workflow**.
6. Watch the live log — the deploy step takes ~2–5 min on first deploy (image pull), ~10–30 s thereafter (cached layers).

After step 5 succeeds, every subsequent `git push` to `main` does this automatically:
*publish-image.yml* → finishes → triggers *deploy.yml* (via `workflow_run`) → fresh container live on the VPS.

### GHCR — making images private (optional)

If you keep the GitHub package **private** instead of public, the deploy workflow needs to log into ghcr.io. Add two more secrets:

| Name         | How to create it                                                                                     |
| ------------ | ---------------------------------------------------------------------------------------------------- |
| `GHCR_USER`  | Your GitHub username (lowercase).                                                                    |
| `GHCR_TOKEN` | A Personal Access Token (classic) with scope **`read:packages`**. github.com → Settings → Developer settings → Personal access tokens → Tokens (classic) → Generate new. |

The deploy workflow auto-detects these and does `docker login ghcr.io` before pulling.

### Opening the dashboard in your browser after the deploy

When the **Deploy to VPS** workflow finishes, its summary (visible at the bottom of the workflow run page) gives you one of two things:

#### Case A — your VPS has a public IP and port 7860/7861 is reachable

The summary shows:

> ### ✅ Dashboard reachable
> Open in your browser:  **http://<VPS_HOST>:<PORT>/**

→ Just click the link, or paste it into any browser. Anyone on the internet who has that URL can open it.

#### Case B — your VPS is behind NAT or its cloud firewall blocks the port

(This is the common case on small VPSs, home servers, or any VPS where only port 22 is forwarded externally.) The summary shows:

> ### ⚠️ Container is healthy, but `<VPS_HOST>:<PORT>` is not reachable from the public internet
>
> Access via SSH tunnel from your laptop:
> ```bash
> ssh -L <PORT>:127.0.0.1:<PORT> <VPS_USER>@<VPS_HOST>
> # then open http://127.0.0.1:<PORT>/
> ```

Concrete example (this is what works for the reference VPS):

```bash
# 1.  In a terminal on YOUR LAPTOP, open the tunnel:
ssh -L 7861:127.0.0.1:7861 \
    -o PreferredAuthentications=password -o PubkeyAuthentication=no \
    novarch3@79.127.114.34

# 2.  Leave that ssh window open.

# 3.  In your browser, open:
#     http://127.0.0.1:7861/
```

(Pick whatever port the workflow actually published — see the summary. Auto-discovery may have shifted it from 7860 → 7861 → 7862 etc. depending on what's already running on the VPS.)

#### Finding the port later, if you've lost the summary

The deploy script always writes the actual host port to `~/<REMOTE_DIR>/.actual-port` on the VPS:

```bash
ssh <VPS_USER>@<VPS_HOST> 'cat ~/physsdb/.actual-port'
```

#### Stopping / restarting the dashboard

**Docker modes (`ghcr` / `build`):**
```bash
ssh <VPS_USER>@<VPS_HOST>
cd ~/physsdb
docker compose --profile gpu down              # stop
docker compose --profile gpu up -d             # start
docker compose --profile gpu logs --tail=50   # live log
```

**Conda mode:**
```bash
ssh <VPS_USER>@<VPS_HOST>
XDG_RUNTIME_DIR=/run/user/$(id -u) systemctl --user status sdb-app
XDG_RUNTIME_DIR=/run/user/$(id -u) systemctl --user restart sdb-app
tail -f ~/physsdb/app.log
```

### When this won't work as-is

| Situation                                                | What to do |
|----------------------------------------------------------|------------|
| VPS only accepts SSH-key auth (PasswordAuthentication no in sshd) | Switch the workflow to key auth — replace `VPS_PASSWORD` with `SSH_PRIVATE_KEY` secret and remove `PreferredAuthentications=password` from the ssh options. |
| VPS user has no sudo                                      | Make the user a sudoer (`usermod -aG sudo $USER`) or use `root`. |
| VPS network blocks Docker Hub / NVIDIA repos              | Build locally and ship the image: `docker save physsdb:gpu \| gzip > img.tar.gz; scp img.tar.gz vps:; ssh vps 'docker load < img.tar.gz'`. |
| Cloud provider firewall blocks port 7860                  | Open it in the provider console (AWS Security Group, DigitalOcean Cloud Firewall, etc.) — the workflow only touches host-level UFW. |
| You need staged deploys (test → prod)                     | Create two GitHub Environments with different secrets, then duplicate the job with `environment: test`. |

### Updating the deployment

Just `git push` to `main`. The workflow detects code changes, rebuilds the image on the VPS (Docker's layer cache makes this fast — usually <60 s if only `app.py` changed), and rolls the container. Active user sessions get a Gradio reconnect popup; they don't lose state in the page.

### Rolling back

`git revert` the bad commit, push to main, the workflow redeploys the previous version automatically.

---

## Citing

If you use PhysSDB in academic work please cite the thesis / paper (TBD)
*and* the MagicBathyNet dataset:

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

---

## License

Code: **MIT** (see [LICENSE](LICENSE)).
Dataset sample tiles: CC-BY-NC-SA 4.0 (MagicBathyNet original license).
Pre-trained weights: MIT — please cite if reused.
