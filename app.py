"""
SDB Research Dashboard — interactive Gradio app comparing a supervised
baseline U-Net against PhysSDB (physics-consistent multi-head model) on
Sentinel-2 imagery.

Run:
    python app.py --host 0.0.0.0 --port 7860

The app loads checkpoints produced by train_sdb.py and train_phys.py:
    runs/{baseline_an, baseline_pl, physsdb_an, physsdb_pl}/model_best.pt
"""
import argparse, io, json, os, sys, tempfile, time
from pathlib import Path

import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import torch
from PIL import Image
from scipy.ndimage import zoom
from skimage import io as skio

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from train_sdb import UNet_bathy, load_norm_param
from train_phys import PhysSDB
from bathy_truth import fetch_truth_chip

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Paths are overridable via env vars so the same code runs in:
#   - the original VPS layout (default)
#   - a Docker container (set SDB_MODEL_ROOT=/app/runs, SDB_DATA_ROOT=/app/data)
#   - a Hugging Face Space (set to local dirs in the Space repo)
RUNS = Path(os.environ.get("SDB_MODEL_ROOT",
                           "/home/novarch2/workspace/sdb_sprint/runs"))
DATA = Path(os.environ.get("SDB_DATA_ROOT",
                           "/home/novarch2/workspace/data/magicbathynet"))

# ---------- model loading ----------------------------------------------------
def load_unet(ckpt, in_channels=3):
    net = UNet_bathy(in_channels=in_channels, out_channels=1).to(DEVICE)
    net.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    net.eval()
    return net

def load_phys(ckpt, norm_depth_abs):
    net = PhysSDB(in_channels=3, norm_depth_abs=norm_depth_abs).to(DEVICE)
    net.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    net.eval()
    return net

MODELS = {
    "Agia Napa (Mediterranean — clear water)": dict(
        base_ckpt=RUNS / "baseline_an/model_best.pt",
        phys_ckpt=RUNS / "physsdb_an/model_best.pt",
        norm_param_path=DATA / "agia_napa/norm_param_s2_an",
        norm_depth=30.443,
        sample_path=DATA / "agia_napa/img/s2/img_411.tif",
        sample_gt_path=DATA / "agia_napa/depth/s2/depth_411.tif",
    ),
    "Puck Lagoon (Baltic — turbid water)": dict(
        base_ckpt=RUNS / "puck_lagoon" if False else RUNS / "baseline_pl/model_best.pt",
        phys_ckpt=RUNS / "physsdb_pl/model_best.pt",
        norm_param_path=DATA / "puck_lagoon/norm_param_s2_pl",
        norm_depth=11.0,
        sample_path=DATA / "puck_lagoon/img/s2/img_411.tif",
        sample_gt_path=DATA / "puck_lagoon/depth/s2/depth_411.tif",
    ),
}

# Lazy load once
_LOADED = {}
def get_models(area_key):
    cfg = MODELS[area_key]
    if area_key not in _LOADED:
        base = load_unet(cfg["base_ckpt"])
        phys = load_phys(cfg["phys_ckpt"], cfg["norm_depth"])
        norm_param = load_norm_param(cfg["norm_param_path"])
        _LOADED[area_key] = (base, phys, norm_param)
    return _LOADED[area_key]

# ---------- preprocessing ----------------------------------------------------
CROP = 256

def preprocess_image(file_path, norm_param):
    img = skio.imread(file_path).astype(np.float32)
    if img.ndim == 2:
        img = img[..., None]
    img = np.transpose(img, (2, 0, 1))[:3]                 # CHW, drop extra bands
    pmin = norm_param[0][:img.shape[0], None, None]
    pmax = norm_param[1][:img.shape[0], None, None]
    rng = np.where((pmax - pmin) > 1e-6, pmax - pmin, 1.0)
    img = np.clip((img - pmin) / rng, 0.0, 1.0)            # → [0,1]
    H, W = img.shape[1], img.shape[2]
    ratio = CROP / max(H, W)
    img_z = zoom(img, (1, ratio, ratio), order=1)
    img_z = img_z[:, :CROP, :CROP]
    if img_z.shape[1] < CROP or img_z.shape[2] < CROP:
        ph = CROP - img_z.shape[1]; pw = CROP - img_z.shape[2]
        img_z = np.pad(img_z, ((0,0),(0,ph),(0,pw)), mode="reflect")
    return img_z.astype(np.float32)

# ---------- inference --------------------------------------------------------
@torch.no_grad()
def run_inference(file_path, area_key):
    base, phys, norm_param = get_models(area_key)
    cfg = MODELS[area_key]
    norm_depth = cfg["norm_depth"]

    img_z = preprocess_image(file_path, norm_param)
    x = torch.from_numpy(img_z).unsqueeze(0).to(DEVICE)

    base_pred_n = base(x).cpu().numpy().squeeze()
    base_pred_m = base_pred_n * norm_depth

    out = phys(x)
    d_n = out["d"].squeeze().cpu().numpy()
    d_m = d_n * norm_depth
    sigma_n = torch.exp(out["log_sigma"]).squeeze().cpu().numpy()
    sigma_m = sigma_n * norm_depth
    a       = out["a"].squeeze().cpu().numpy()
    b_b     = out["b_b"].squeeze().cpu().numpy()
    rho     = out["rho"].squeeze().cpu().numpy()                  # [3,H,W]
    R_hat   = out["R_hat"].squeeze().cpu().numpy()                # [3,H,W]
    R_obs   = img_z                                                # [3,H,W]
    residual = np.linalg.norm(R_hat - R_obs, axis=0)              # [H,W]

    # optional: ground-truth if a sample tile (helpful for "comparison" rows)
    gt_m = None
    sample_gt = cfg.get("sample_gt_path")
    try:
        if sample_gt and Path(sample_gt).exists() and Path(file_path).name == Path(cfg["sample_path"]).name:
            g = skio.imread(sample_gt).astype(np.float32)
            ratio = CROP / max(g.shape[0], g.shape[1])
            g = zoom(g, (ratio, ratio), order=1)[:CROP, :CROP]
            if g.shape[0] < CROP or g.shape[1] < CROP:
                ph = CROP - g.shape[0]; pw = CROP - g.shape[1]
                g = np.pad(g, ((0,ph),(0,pw)), mode="reflect")
            gt_m = np.abs(g)              # absolute depth in meters
    except Exception:
        gt_m = None

    return dict(
        rgb_obs=img_z, rgb_hat=R_hat, residual=residual,
        base_pred_m=base_pred_m, phys_pred_m=d_m, sigma_m=sigma_m,
        a=a, b_b=b_b, rho=rho, gt_m=gt_m,
        norm_depth=norm_depth, area=area_key,
    )

# ---------- inference on an already-prepared chip (used by Acquisition tab) --
@torch.no_grad()
def run_inference_chip(chip_chw_01, area_key, sample_gt_path=None):
    """chip_chw_01: float32 (3,H,W) Sentinel-2 surface reflectance in [0,1].
    Applies the area's MagicBathyNet norm_param to bring values into the
    distribution the models were trained on, then runs both networks."""
    base, phys, norm_param = get_models(area_key)
    cfg = MODELS[area_key]
    norm_depth = cfg["norm_depth"]

    # The MagicBathyNet S2 tiles are stored at L2A DN scale (~0..10000).
    # Our chip is reflectance ([0,1]) → multiply by 10000 to match training distribution.
    chip_dn = chip_chw_01.astype(np.float32) * 10000.0
    img = chip_dn[:3]
    pmin = norm_param[0][:img.shape[0], None, None]
    pmax = norm_param[1][:img.shape[0], None, None]
    rng = np.where((pmax - pmin) > 1e-6, pmax - pmin, 1.0)
    img_z = np.clip((img - pmin) / rng, 0.0, 1.0).astype(np.float32)
    # ensure CROPxCROP shape (already 256x256 in our acquired chips)
    if img_z.shape[1] != CROP or img_z.shape[2] != CROP:
        ph = CROP - img_z.shape[1]; pw = CROP - img_z.shape[2]
        img_z = np.pad(img_z, ((0,0),(0,max(0,ph)),(0,max(0,pw))), mode="reflect")
        img_z = img_z[:, :CROP, :CROP]

    x = torch.from_numpy(img_z).unsqueeze(0).to(DEVICE)
    base_pred_n = base(x).cpu().numpy().squeeze()
    base_pred_m = base_pred_n * norm_depth
    out = phys(x)
    d_n = out["d"].squeeze().cpu().numpy()
    d_m = d_n * norm_depth
    sigma_m = torch.exp(out["log_sigma"]).squeeze().cpu().numpy() * norm_depth
    a   = out["a"].squeeze().cpu().numpy()
    b_b = out["b_b"].squeeze().cpu().numpy()
    rho = out["rho"].squeeze().cpu().numpy()
    R_hat = out["R_hat"].squeeze().cpu().numpy()
    residual = np.linalg.norm(R_hat - img_z, axis=0)

    return dict(
        rgb_obs=img_z, rgb_hat=R_hat, residual=residual,
        base_pred_m=base_pred_m, phys_pred_m=d_m, sigma_m=sigma_m,
        a=a, b_b=b_b, rho=rho, gt_m=None,
        norm_depth=norm_depth, area=area_key,
    )

# ---------- visualization helpers --------------------------------------------
def to_rgb_image(rgb_chw, gamma=0.7):
    """Display-friendly RGB from CHW [0,1] array (mild gamma to brighten)."""
    rgb = np.transpose(rgb_chw, (1, 2, 0))
    rgb = np.clip(rgb ** gamma, 0, 1)
    return (rgb * 255).astype(np.uint8)

def heatmap_rgb(arr, vmin=None, vmax=None, cmap="viridis_r"):
    a = np.asarray(arr, dtype=np.float32)
    if vmin is None: vmin = float(np.nanmin(a))
    if vmax is None: vmax = float(np.nanmax(a))
    if vmax <= vmin: vmax = vmin + 1e-6
    a = (a.clip(vmin, vmax) - vmin) / (vmax - vmin)
    cmap_obj = cm.get_cmap(cmap)
    rgb = cmap_obj(a)[..., :3]
    return (rgb * 255).astype(np.uint8)

# ---------- labeled-figure layout (used for click-coord remapping) -----------
FIG_W_PX, FIG_H_PX = 520, 460                        # final PNG size at 100 dpi
DATA_AX_RECT = (0.04, 0.06, 0.74, 0.84)              # [left, bottom, w, h] in figure fraction
DATA_X0 = int(DATA_AX_RECT[0] * FIG_W_PX)
# pixel y origin (top-left): convert matplotlib bottom-origin fraction → pixel-from-top
DATA_Y0 = int((1 - DATA_AX_RECT[1] - DATA_AX_RECT[3]) * FIG_H_PX)
DATA_W  = int(DATA_AX_RECT[2] * FIG_W_PX)
DATA_H  = int(DATA_AX_RECT[3] * FIG_H_PX)

def labeled_heatmap(arr, vmin, vmax, cmap, title, unit_label):
    """Render an array as a heatmap with a colorbar, title, and unit label.
    Output PNG dimensions are constant so that pixel clicks can be remapped
    back to the source array via (DATA_X0, DATA_Y0, DATA_W, DATA_H)."""
    fig = plt.figure(figsize=(FIG_W_PX / 100, FIG_H_PX / 100), dpi=100,
                     facecolor="white")
    ax = fig.add_axes(DATA_AX_RECT)
    im = ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax,
                   aspect="auto", interpolation="nearest")
    ax.set_title(title, fontsize=11, pad=8)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_visible(False)
    cax = fig.add_axes([DATA_AX_RECT[0] + DATA_AX_RECT[2] + 0.04,
                        DATA_AX_RECT[1], 0.04, DATA_AX_RECT[3]])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label(unit_label, fontsize=10)
    cb.ax.tick_params(labelsize=9)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return np.array(Image.open(buf).convert("RGB"))

def labeled_rgb(rgb_chw, title, subtitle=""):
    """Render an RGB CHW [0,1] array with title (no colorbar). Same dims as labeled_heatmap."""
    rgb = np.transpose(rgb_chw, (1, 2, 0))
    rgb = np.clip(rgb ** 0.7, 0, 1)
    fig = plt.figure(figsize=(FIG_W_PX / 100, FIG_H_PX / 100), dpi=100,
                     facecolor="white")
    ax = fig.add_axes(DATA_AX_RECT)
    ax.imshow(rgb, aspect="auto", interpolation="nearest")
    ax.set_title(title + ("\n" + subtitle if subtitle else ""),
                 fontsize=11, pad=8)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_visible(False)
    # transparent "ghost" axes where the colorbar would go, so layout matches heatmaps
    cax = fig.add_axes([DATA_AX_RECT[0] + DATA_AX_RECT[2] + 0.04,
                        DATA_AX_RECT[1], 0.04, DATA_AX_RECT[3]])
    cax.axis("off")
    cax.text(0.5, 0.5, "RGB",
             rotation=90, ha="center", va="center", fontsize=9, color="#888",
             transform=cax.transAxes)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return np.array(Image.open(buf).convert("RGB"))

def displayed_xy_to_data(cx, cy, data_shape=(256, 256)):
    """Convert a click on a labeled figure back to (x, y) in the source data array."""
    H, W = data_shape
    fx = (cx - DATA_X0) / DATA_W
    fy = (cy - DATA_Y0) / DATA_H
    if not (0.0 <= fx < 1.0 and 0.0 <= fy < 1.0):
        return None
    return int(fx * W), int(fy * H)

def add_colorbar_strip(rgb_arr, vmin, vmax, cmap="viridis_r", label="m"):
    """Concatenate a thin colorbar strip on the right (visual only, not clickable)."""
    H, W, _ = rgb_arr.shape
    bar_w = 18
    bar = np.linspace(0, 1, H)[::-1, None].repeat(bar_w, axis=1)
    cmap_obj = cm.get_cmap(cmap)
    bar_rgb = (cmap_obj(bar)[..., :3] * 255).astype(np.uint8)
    # text labels (vmin/vmax) drawn via matplotlib for crispness
    fig, ax = plt.subplots(figsize=(0.3, H / 100), dpi=100)
    ax.imshow(bar, aspect="auto", cmap=cmap)
    ax.set_xticks([]); ax.set_yticks([0, H - 1])
    ax.set_yticklabels([f"{vmax:.1f}", f"{vmin:.1f}"])
    ax.set_ylabel(label)
    fig.tight_layout(pad=0)
    buf = io.BytesIO(); fig.savefig(buf, format="png", bbox_inches="tight"); plt.close(fig)
    buf.seek(0)
    bar_img = np.array(Image.open(buf).convert("RGB").resize((40, H)))
    return np.concatenate([rgb_arr, bar_img], axis=1)

# ---------- shared post-processing helper ------------------------------------
def _build_outputs(res, extra_table_rows=None):
    # shared depth color scale so the two depth maps compare directly
    vmin_d = 0.0
    vmax_d = float(np.nanpercentile(np.concatenate([res["base_pred_m"].ravel(),
                                                    res["phys_pred_m"].ravel()]), 99))
    vmax_d = max(vmax_d, 1.0)
    smax = float(np.nanpercentile(res["sigma_m"], 99)); smax = max(smax, 1e-3)
    rmax = float(np.nanpercentile(res["residual"], 99)); rmax = max(rmax, 1e-3)
    area = res.get("area", "")

    rgb_in_img  = labeled_rgb(res["rgb_obs"],
                              title="Input image — preprocessed RGB",
                              subtitle=f"{area}  ·  256 × 256 px")
    base_img    = labeled_heatmap(res["base_pred_m"], vmin_d, vmax_d, "viridis_r",
                                  title="Baseline U-Net — predicted depth",
                                  unit_label="depth (m)")
    phys_img    = labeled_heatmap(res["phys_pred_m"], vmin_d, vmax_d, "viridis_r",
                                  title="PhysSDB — predicted depth  [click to inspect a pixel]",
                                  unit_label="depth (m)")
    sigma_img   = labeled_heatmap(res["sigma_m"], 0.0, smax, "magma",
                                  title="PhysSDB — per-pixel uncertainty  σ",
                                  unit_label="σ (m, 1 std)")
    rgb_hat_img = labeled_rgb(res["rgb_hat"],
                              title="PhysSDB — reconstructed RGB  (R̂ from Lee-1998)",
                              subtitle="should resemble input where physics fits")
    res_img     = labeled_heatmap(res["residual"], 0.0, rmax, "inferno",
                                  title="PhysSDB — reconstruction residual  ‖R̂ − R_obs‖",
                                  unit_label="residual (rel. reflectance)")

    # Measured-bathymetry comparison panels (only if a reference truth is attached)
    gt = res.get("gt_m")
    have_truth = gt is not None and np.isfinite(gt).any()
    n_in_range = 0; n_too_deep = 0; n_water = 0
    if have_truth:
        gt_arr = np.asarray(gt, dtype=np.float32)
        # The trained model can only predict depths up to its training range
        # (norm_depth_abs). Pixels deeper than that are out-of-distribution and
        # would dominate the error stats; mask them as "land" for the comparison.
        train_max_m = float(res["norm_depth"])
        is_water = np.isfinite(gt_arr) & (gt_arr > 0)
        in_range = is_water & (gt_arr <= train_max_m)
        too_deep = is_water & (gt_arr > train_max_m)
        n_water = int(is_water.sum())
        n_in_range = int(in_range.sum())
        n_too_deep = int(too_deep.sum())

        # Truth panel: show actual depths (so you can SEE the open-water trench),
        # but cap the colorbar so coastal detail isn't washed out
        truth_vmax_display = max(train_max_m, float(vmax_d))
        gt_img = labeled_heatmap(np.where(is_water, gt_arr, np.nan),
                                 0.0, truth_vmax_display, "viridis_r",
                                 title=(f"Measured depth — {res.get('gt_source','reference')}\n"
                                        f"({n_water} water px, {n_too_deep} deeper than "
                                        f"model's training max {train_max_m:.0f} m)"),
                                 unit_label="depth (m)")
        # Per-method absolute error, restricted to pixels INSIDE the training range
        err_b = np.where(in_range, np.abs(res["base_pred_m"] - gt_arr), np.nan)
        err_p = np.where(in_range, np.abs(res["phys_pred_m"] - gt_arr), np.nan)
        fb = err_b[np.isfinite(err_b)]; fp = err_p[np.isfinite(err_p)]
        if fb.size + fp.size > 0:
            emax = float(np.nanpercentile(np.concatenate([fb, fp]), 95))
        else:
            emax = 1.0
        emax = max(emax, 0.5)
        err_b_img = labeled_heatmap(err_b, 0.0, emax, "inferno",
                                    title=(f"Baseline error  |baseline − measured|\n"
                                           f"(only pixels with measured depth ≤ {train_max_m:.0f} m)"),
                                    unit_label="abs depth error (m)")
        err_p_img = labeled_heatmap(err_p, 0.0, emax, "inferno",
                                    title=(f"PhysSDB error  |PhysSDB − measured|\n"
                                           f"(only pixels with measured depth ≤ {train_max_m:.0f} m)"),
                                    unit_label="abs depth error (m)")
        # stash the masks so the table and click handler can use them
        res["_in_range_mask"] = in_range
        res["_train_max_m"]   = train_max_m
        res["_n_water"] = n_water
        res["_n_too_deep"] = n_too_deep
    else:
        gt_img = err_b_img = err_p_img = None

    # comparison table
    rows = [
        ["Mean predicted depth (m)", f"{res['base_pred_m'].mean():.2f}",
                                     f"{res['phys_pred_m'].mean():.2f}"],
        ["Median predicted depth (m)", f"{np.median(res['base_pred_m']):.2f}",
                                       f"{np.median(res['phys_pred_m']):.2f}"],
        ["Predicted depth range (m)",
         f"[{res['base_pred_m'].min():.2f}, {res['base_pred_m'].max():.2f}]",
         f"[{res['phys_pred_m'].min():.2f}, {res['phys_pred_m'].max():.2f}]"],
        ["Per-pixel σ (m)", "— (not produced by baseline)",
                            f"mean {res['sigma_m'].mean():.2f}, median {np.median(res['sigma_m']):.2f}"],
        ["Physics reconstruction MAE (rel. units)", "—",
                            f"{res['residual'].mean():.3f}"],
    ]
    # add GT-comparison rows if available
    if res.get("gt_m") is not None:
        g = np.asarray(res["gt_m"], dtype=np.float32)
        m_in = res.get("_in_range_mask")
        train_max_m = res.get("_train_max_m", float(res["norm_depth"]))
        n_water = res.get("_n_water", 0)
        n_too_deep = res.get("_n_too_deep", 0)
        src = res.get("gt_source", "reference")
        rows.append([f"Water pixels found ({src})",
                     f"{n_water} / {int(g.size)}",
                     f"({100*n_water/max(1,g.size):.1f}% of chip)"])
        rows.append([f"Out-of-training-range water pixels",
                     f"{n_too_deep}  (measured > {train_max_m:.0f} m, model can't predict these)",
                     "excluded from RMSE/MAE/bias"])
        if m_in is not None and m_in.sum() > 0:
            be = res["base_pred_m"][m_in] - g[m_in]
            pe = res["phys_pred_m"][m_in] - g[m_in]
            rows.append([f"RMSE vs measured (m, in-range only)",
                         f"{float(np.sqrt(np.mean(be**2))):.3f}",
                         f"{float(np.sqrt(np.mean(pe**2))):.3f}"])
            rows.append([f"MAE vs measured (m, in-range only)",
                         f"{float(np.mean(np.abs(be))):.3f}",
                         f"{float(np.mean(np.abs(pe))):.3f}"])
            rows.append([f"Bias vs measured (m, in-range only)",
                         f"{float(np.mean(be)):+.3f}",
                         f"{float(np.mean(pe)):+.3f}"])
        else:
            rows.append([f"RMSE/MAE/bias", "—",
                         "no in-range water pixels to compare"])
    if extra_table_rows:
        rows = extra_table_rows + rows

    # save outputs to a temp dir for download
    tmp = Path(tempfile.mkdtemp(prefix="sdb_dash_"))
    np.savez(tmp / "predictions.npz",
             input_rgb_chw=res["rgb_obs"],
             baseline_depth_m=res["base_pred_m"],
             physsdb_depth_m=res["phys_pred_m"],
             physsdb_sigma_m=res["sigma_m"],
             physsdb_R_hat_chw=res["rgb_hat"],
             physsdb_residual=res["residual"],
             physsdb_a=res["a"], physsdb_b_b=res["b_b"],
             physsdb_rho_chw=res["rho"],
             area=res["area"], norm_depth=res["norm_depth"])
    download_files = [str(tmp / "predictions.npz")]

    # serialize the prediction state for pixel-click handlers (NaN→None for JSON)
    gt_for_state = None
    if res.get("gt_m") is not None:
        gt_arr = np.asarray(res["gt_m"], dtype=np.float32)
        gt_for_state = np.where(np.isfinite(gt_arr), gt_arr, np.nan).tolist()
    state = dict(
        base_pred_m=res["base_pred_m"].tolist(),
        phys_pred_m=res["phys_pred_m"].tolist(),
        sigma_m=res["sigma_m"].tolist(),
        a=res["a"].tolist(),
        b_b=res["b_b"].tolist(),
        rho=res["rho"].tolist(),
        residual=res["residual"].tolist(),
        rgb_obs=res["rgb_obs"].tolist(),
        rgb_hat=res["rgb_hat"].tolist(),
        gt_m=gt_for_state,
        gt_source=res.get("gt_source"),
        train_max_m=res.get("_train_max_m") or res["norm_depth"],
        area=res["area"],
        norm_depth=res["norm_depth"],
    )

    color_msg = (f"Depth color scale: 0 → {vmax_d:.1f} m  (shared between baseline & PhysSDB)"
                 + ("  ·  measured-depth panel uses NOAA NGDC mosaic"
                    if have_truth else ""))
    return (rgb_in_img, base_img, phys_img, sigma_img,
            rgb_hat_img, res_img, gt_img, err_b_img, err_p_img,
            rows, download_files, state, color_msg)

# ---------- upload-tab driver ------------------------------------------------
def analyse(file, area_key):
    if file is None:
        raise gr.Error("Please upload a Sentinel-2 image first (or click 'Load sample').")
    file_path = file if isinstance(file, str) else file.name
    res = run_inference(file_path, area_key)
    return _build_outputs(res)

# ---------- acquisition-tab drivers (3-step flow) ----------------------------
def _validate_bbox(lat_min, lat_max, lon_min, lon_max):
    vals = [lat_min, lat_max, lon_min, lon_max]
    if any(v is None or v == "" for v in vals):
        raise gr.Error("Draw a rectangle on the map first (▭ tool top-left), or fill the 4 bbox boxes manually.")
    try:
        lat_min, lat_max, lon_min, lon_max = map(float, vals)
    except Exception:
        raise gr.Error("Bbox values must be numeric.")
    if lat_min >= lat_max or lon_min >= lon_max:
        raise gr.Error("Bbox invalid: south/west must be smaller than north/east.")
    if not (-90 <= lat_min and lat_max <= 90 and -180 <= lon_min and lon_max <= 180):
        raise gr.Error("Bbox lat/lon out of range.")
    return lat_min, lat_max, lon_min, lon_max


def fetch_preview(lat_min, lat_max, lon_min, lon_max,
                  days_back, max_cloud, progress=gr.Progress()):
    """Step 2: search the S3 bucket for the best scene covering the drawn bbox,
    download a chip, return a preview image + the chip stashed in state for the
    next step.

    Returns:
      preview_rgb_uint8, meta_markdown, chip_state_dict, log_markdown, enable_analyse_button
    """
    import math
    import s2_acquire as s2a

    lat_min, lat_max, lon_min, lon_max = _validate_bbox(lat_min, lat_max, lon_min, lon_max)
    lat_c = (lat_min + lat_max) / 2
    lon_c = (lon_min + lon_max) / 2
    # bbox extent in meters (max of width/height; we'll fetch a square chip covering the AOI)
    width_m  = abs(lon_max - lon_min) * 111_320.0 * math.cos(math.radians(lat_c))
    height_m = abs(lat_max - lat_min) * 111_320.0
    size_m = max(width_m, height_m)
    if size_m < 1000.0:
        size_m = 1000.0                      # at least 1 km × 1 km of S2 imagery
    if size_m > 30000.0:
        size_m = 30000.0                     # cap at 30 km

    log_lines = []
    def cb(msg):
        log_lines.append(msg)
        try: progress(0.5, desc=msg[:90])
        except Exception: pass

    progress(0.05, desc="searching sentinel-cogs S3 for low-cloud scene…")
    chip, info, _log = s2a.search_and_download(
        lat=lat_c, lon=lon_c,
        days_back=int(days_back),
        max_cloud=float(max_cloud),
        size_px=CROP,
        size_m=size_m,
        progress_cb=cb,
    )
    log_md = "### Search log\n```\n" + "\n".join(log_lines) + "\n```"
    if chip is None:
        raise gr.Error("No Sentinel-2 scene matched the filter. Try a larger date window or higher cloud cover.")

    scene = info["scene"]
    # also fetch reference bathymetry truth for the *same* bbox
    progress(0.85, desc="fetching reference bathymetry (NOAA DEM Global Mosaic)…")
    truth_depth_m, truth_info = fetch_truth_chip(lat_min, lat_max, lon_min, lon_max,
                                                 size_px=CROP)
    if truth_depth_m is not None:
        log_lines.append(f"truth: water {truth_info['water_pixels']}/{truth_info['total_pixels']} px "
                         f"({100*truth_info['water_fraction']:.0f}%), depth range "
                         f"{truth_info['depth_min_m']:.2f}–{truth_info['depth_max_m']:.2f} m")
    else:
        log_lines.append(f"truth: fetch failed — {truth_info.get('error','?')}")
    log_md = "### Search log\n```\n" + "\n".join(log_lines) + "\n```"
    preview = labeled_rgb(chip,
                          title=f"Fetched scene — {scene['scene_id']}",
                          subtitle=f"acq {scene['acq_date']}  ·  cloud {scene['cloud_cover']:.1f}%  ·  "
                                   f"chip {info['extent_m']:.0f} m, {info['m_per_pixel']:.2f} m/px")
    meta_md = (
        f"### Selected Sentinel-2 scene\n\n"
        f"- **Scene ID**: `{scene['scene_id']}`\n"
        f"- **Acquisition date**: {scene['acq_date']}\n"
        f"- **Cloud cover**: {scene['cloud_cover']:.1f} %\n"
        f"- **Tile data coverage**: {scene['data_coverage']:.1f} %\n"
        f"- **MGRS grid**: {scene['utm_zone']}{scene['lat_band']}{scene['grid_id']}  (UTM EPSG:{info['utm_epsg']})\n"
        f"- **Chip extent**: {info['extent_m']:.0f} m × {info['extent_m']:.0f} m, "
        f"{info['size_px']}×{info['size_px']} px  ({info['m_per_pixel']:.2f} m/px)\n"
        f"- **Center**: ({info['center_lat']:.4f}, {info['center_lon']:.4f})\n"
        f"- **Chip stats**: min={info['chip_min']:.3f}  mean={info['chip_mean']:.3f}  max={info['chip_max']:.3f}"
    )
    state = {"chip": chip.tolist(), "info": info,
             "truth": truth_depth_m.tolist() if truth_depth_m is not None else None,
             "truth_info": truth_info}
    progress(1.0, desc="preview ready")
    return preview, meta_md, state, log_md, gr.update(interactive=True)


def analyse_preview(chip_state, area_key, progress=gr.Progress()):
    """Step 3: run baseline + PhysSDB on the chip stashed by the preview step,
    and compare to NOAA reference bathymetry where available."""
    if not chip_state or "chip" not in chip_state:
        raise gr.Error("Fetch a preview first (🔎 button).")
    chip = np.asarray(chip_state["chip"], dtype=np.float32)
    info = chip_state["info"]
    progress(0.1, desc="running both models on the chip…")
    res = run_inference_chip(chip, area_key)
    # Attach the NOAA-truth depth (positive meters, NaN over land) for comparison
    if chip_state.get("truth") is not None:
        res["gt_m"] = np.asarray(chip_state["truth"], dtype=np.float32)
        res["gt_source"] = "NOAA NGDC DEM Global Mosaic"
    scene = info["scene"]
    extra = [
        ["Sentinel-2 scene", scene["scene_id"],
         f"acq {scene['acq_date']}, cloud {scene['cloud_cover']:.1f}%, cov {scene['data_coverage']:.1f}%"],
        ["MGRS grid", f"{scene['utm_zone']}{scene['lat_band']}{scene['grid_id']}",
         f"UTM EPSG:{info['utm_epsg']}"],
        ["Chip center (lat, lon)", f"{info['center_lat']:.4f}, {info['center_lon']:.4f}",
         f"extent {info['extent_m']:.0f} m  ({info['m_per_pixel']:.2f} m/px)"],
        ["Chip stats", f"min={info['chip_min']:.3f}",
         f"max={info['chip_max']:.3f}  mean={info['chip_mean']:.3f}"],
    ]
    progress(1.0, desc="done")
    return _build_outputs(res, extra_table_rows=extra)

# ---------- pixel-click handler ---------------------------------------------
def on_pixel_click(state, evt: gr.SelectData):
    if state is None:
        return "Run an analysis first."
    cx, cy = evt.index[0], evt.index[1]
    H = len(state["phys_pred_m"]); W = len(state["phys_pred_m"][0])
    mapped = displayed_xy_to_data(cx, cy, data_shape=(H, W))
    if mapped is None:
        return (f"*Clicked outside the data area (px {cx}, {cy}). "
                f"Click inside the colored region of the depth map.*")
    x, y = mapped
    b = state["base_pred_m"][y][x]
    p = state["phys_pred_m"][y][x]
    s = state["sigma_m"][y][x]
    a = state["a"][y][x]
    bb = state["b_b"][y][x]
    rR, rG, rB = state["rho"][0][y][x], state["rho"][1][y][x], state["rho"][2][y][x]
    res_v = state["residual"][y][x]
    obs_r = state["rgb_obs"][0][y][x]; obs_g = state["rgb_obs"][1][y][x]; obs_b = state["rgb_obs"][2][y][x]
    hat_r = state["rgb_hat"][0][y][x]; hat_g = state["rgb_hat"][1][y][x]; hat_b = state["rgb_hat"][2][y][x]
    # measured depth (may be missing or NaN over land or beyond training range)
    gt = state.get("gt_m")
    gt_v = gt[y][x] if gt is not None else None
    gt_src = state.get("gt_source", "reference")
    train_max = state.get("train_max_m") or state.get("norm_depth")
    is_water = (gt_v is not None and isinstance(gt_v, (int, float)) and gt_v == gt_v
                and gt_v > 0)
    if gt_v is None:
        meas_line   = "| **Measured depth (m)** | — | — |"
        meas_note   = "*(no reference depth loaded)*"
        err_lines = ""
    elif not is_water:
        meas_line   = "| **Measured depth (m)** | — | — |"
        meas_note   = "*(land / no-data at this pixel)*"
        err_lines = ""
    else:
        meas_line = f"| **Measured depth (m)** | {gt_v:.3f} | {gt_v:.3f} |"
        meas_note = f"*source: {gt_src}*"
        if train_max is not None and gt_v > float(train_max):
            err_lines = (f"\n> ⚠️ Measured depth **{gt_v:.1f} m** exceeds the model's "
                         f"training range (0 – {float(train_max):.0f} m). Predictions in "
                         f"this regime are extrapolation — the absolute error below is "
                         f"expected to be large and is NOT a fair test of the method.\n\n"
                         f"| **Error vs measured (m)** | {b - gt_v:+.3f} | {p - gt_v:+.3f} |\n"
                         f"| **Absolute error (m)** | {abs(b - gt_v):.3f} | {abs(p - gt_v):.3f} |\n")
        else:
            err_lines = (f"| **Error vs measured (m)** | {b - gt_v:+.3f} | {p - gt_v:+.3f} |\n"
                         f"| **Absolute error (m)** | {abs(b - gt_v):.3f} | {abs(p - gt_v):.3f} |\n"
                         f"| **Abs-error / σ  (PhysSDB only)** | — | {abs(p - gt_v) / max(s, 1e-6):.2f}  "
                         f"*(< 1.96 ≈ within 95% PI)* |\n")
    md = f"""
### Pixel ({x}, {y})  — area: *{state['area']}*

| Field | Baseline | PhysSDB |
|---|---:|---:|
| **Predicted depth (m)** | {b:.3f} | {p:.3f} |
| **Predicted σ (m)** | — | {s:.3f} |
| **95 % PI (m)** | — | [{max(0,p-1.96*s):.2f}, {p+1.96*s:.2f}] |
{meas_line}
{err_lines}

{meas_note}

**PhysSDB predicted optical parameters**

| Parameter | Value |
|---|---:|
| Absorption  *a*       | {a:.4f} |
| Backscatter *b_b*    | {bb:.4f} |
| Bottom reflectance ρ_R | {rR:.4f} |
| Bottom reflectance ρ_G | {rG:.4f} |
| Bottom reflectance ρ_B | {rB:.4f} |
| Reconstruction residual (||R̂−R_obs||) | {res_v:.4f} |

**Reflectance check (input vs PhysSDB reconstruction):**

| Band | Observed | Reconstructed (R̂) | abs error |
|---|---:|---:|---:|
| Red   | {obs_r:.4f} | {hat_r:.4f} | {abs(obs_r-hat_r):.4f} |
| Green | {obs_g:.4f} | {hat_g:.4f} | {abs(obs_g-hat_g):.4f} |
| Blue  | {obs_b:.4f} | {hat_b:.4f} | {abs(obs_b-hat_b):.4f} |
"""
    return md

# ---------- sample-loader handler --------------------------------------------
def load_sample(area_key):
    p = MODELS[area_key]["sample_path"]
    if not Path(p).exists():
        raise gr.Error(f"Sample tile not found at {p}")
    return str(p)

# ---------- Gradio interface --------------------------------------------------
INSTRUCTIONS = """
# 🌊 Satellite-Derived Bathymetry — interactive research dashboard

Compare **Baseline U-Net** (pure data-driven) against **PhysSDB**
(physics-consistent, with differentiable Lee-1998 radiative-transfer self-supervision
and per-pixel uncertainty) on Sentinel-2 imagery.

### How to prepare your input

| Requirement | Detail |
|---|---|
| **File format** | GeoTIFF (`.tif`/`.tiff`). PNG/JPEG also accepted for quick demos. |
| **Channels** | 3 bands in order **Red (B4), Green (B3), Blue (B2)** — i.e. native Sentinel-2 visible RGB. |
| **Image size** | Anything ≤ 256 × 256 ideal; larger imagery is resampled to 256 × 256 (one ROI at a time). |
| **Radiometric scale** | Surface-reflectance DN values that match MagicBathyNet (≈ 0–2500). Pre-trained checkpoints were trained on **MagicBathyNet** Sentinel-2 L2A patches; out-of-distribution inputs may produce unreliable depths. |
| **Easiest path** | Click **"Load sample tile"** below to use a tile from the MagicBathyNet test split. |

### What the models output

- **Baseline U-Net** → a single per-pixel depth in meters. No uncertainty.
- **PhysSDB** → per-pixel **depth**, **σ (Gaussian uncertainty)**, plus the predicted optical-water parameters
  (absorption *a*, backscatter *b_b*, per-band bottom reflectance *ρ_R/G/B*), and a **physics-reconstructed RGB image**
  obtained by passing those parameters through a simplified Lee-1998 radiative-transfer model.
  → low reconstruction residual = physics fits → trust the depth more.
  → high reconstruction residual = physics breaks (turbid / Case-2 water) → trust less.

Click any pixel on the **PhysSDB depth** map to inspect the full predicted parameter vector at that pixel.
"""

LEAFLET_MAP_HTML = """
<div id="sdb-map" style="height:460px;width:100%;border-radius:8px;border:1px solid #ccc;background:#eef"></div>
<div id="sdb-map-status" style="font-size:12px;color:#777;margin-top:4px">
  Loading map… use the rectangle tool (top-left toolbar) to draw an area of interest over coastal water.
</div>
"""

# Injected via demo.load(js=...) so Gradio doesn't sanitize away the <script>.
# Loads Leaflet from CDN dynamically, then initializes the map and wires
# click → reactive update of the lat/lon Gradio Number inputs.
LEAFLET_INIT_JS = r"""
() => {
  if (window.__sdbMapInit) return [];
  window.__sdbMapInit = true;

  function loadCss(href) {
    const l = document.createElement('link');
    l.rel = 'stylesheet';
    l.href = href;
    document.head.appendChild(l);
  }
  function loadScript(src) {
    return new Promise((res, rej) => {
      const s = document.createElement('script');
      s.src = src;
      s.onload = res;
      s.onerror = rej;
      document.head.appendChild(s);
    });
  }
  function reactSet(elemId, val) {
    const el = document.querySelector('#' + elemId + ' input, #' + elemId + ' textarea');
    if (!el) return;
    const proto = el.tagName === 'TEXTAREA'
      ? window.HTMLTextAreaElement.prototype
      : window.HTMLInputElement.prototype;
    const setter = Object.getOwnPropertyDescriptor(proto, 'value').set;
    setter.call(el, val);
    el.dispatchEvent(new Event('input',  { bubbles: true }));
    el.dispatchEvent(new Event('change', { bubbles: true }));
  }

  loadCss('https://unpkg.com/leaflet@1.9.4/dist/leaflet.css');
  loadCss('https://unpkg.com/leaflet-draw@1.0.4/dist/leaflet.draw.css');
  loadScript('https://unpkg.com/leaflet@1.9.4/dist/leaflet.js').then(() =>
    loadScript('https://unpkg.com/leaflet-draw@1.0.4/dist/leaflet.draw.js')
  ).then(() => {
    function tryInit() {
      const el = document.getElementById('sdb-map');
      if (!el || typeof L === 'undefined' || !L.Control.Draw) return setTimeout(tryInit, 200);
      if (window.__sdbMapObj) return;
      const startLat = 34.99, startLng = 34.00;
      const map = L.map('sdb-map').setView([startLat, startLng], 8);
      L.tileLayer('https://a.tile.openstreetmap.fr/hot/{z}/{x}/{y}.png', {
        attribution: '© OpenStreetMap contributors, HOT',
        maxZoom: 18,
      }).addTo(map);

      // Drawing toolbar — only rectangle enabled
      const drawnItems = new L.FeatureGroup();
      map.addLayer(drawnItems);
      const drawControl = new L.Control.Draw({
        position: 'topleft',
        draw: {
          polyline: false, polygon: false, circle: false,
          marker: false, circlemarker: false,
          rectangle: { shapeOptions: { color: '#3388ff', weight: 2, fillOpacity: 0.1 } }
        },
        edit: { featureGroup: drawnItems, edit: false, remove: true }
      });
      map.addControl(drawControl);

      function reportBounds(b) {
        const sw = b.getSouthWest(), ne = b.getNorthEast();
        reactSet('sdb_lat_min', sw.lat.toFixed(5));
        reactSet('sdb_lat_max', ne.lat.toFixed(5));
        reactSet('sdb_lon_min', sw.lng.toFixed(5));
        reactSet('sdb_lon_max', ne.lng.toFixed(5));
        const wkm = (ne.lng - sw.lng) * 111.32 * Math.cos(((sw.lat + ne.lat) / 2) * Math.PI / 180);
        const hkm = (ne.lat - sw.lat) * 111.32;
        const status = document.getElementById('sdb-map-status');
        if (status) status.innerText =
          'Selected area: ' + Math.abs(wkm).toFixed(2) + ' × ' + Math.abs(hkm).toFixed(2) + ' km   '
          + 'center (' + ((sw.lat+ne.lat)/2).toFixed(4) + ', ' + ((sw.lng+ne.lng)/2).toFixed(4) + ').';
      }
      map.on(L.Draw.Event.CREATED, (e) => {
        drawnItems.clearLayers();
        drawnItems.addLayer(e.layer);
        reportBounds(e.layer.getBounds());
      });
      map.on(L.Draw.Event.DELETED, () => {
        ['sdb_lat_min','sdb_lat_max','sdb_lon_min','sdb_lon_max'].forEach(k => reactSet(k, ''));
        const status = document.getElementById('sdb-map-status');
        if (status) status.innerText = 'Use the rectangle tool to draw an area of interest.';
      });

      const status = document.getElementById('sdb-map-status');
      if (status) status.innerText =
        'Map ready. Click the rectangle icon (top-left) and drag to draw an area of interest over coastal water.';
      window.__sdbMapObj = map;
      window.__sdbDrawnItems = drawnItems;
      // Tab-switch resize fix
      setTimeout(() => map.invalidateSize(), 600);
      document.querySelectorAll('[role="tab"]').forEach(t => {
        t.addEventListener('click', () => setTimeout(() => map.invalidateSize(), 250));
      });
    }
    tryInit();
  }).catch((e) => {
    const status = document.getElementById('sdb-map-status');
    if (status) status.innerText = 'Failed to load Leaflet from unpkg.com: ' + e;
  });
  return [];
}
"""

with gr.Blocks(title="SDB Research Dashboard") as demo:
    gr.Markdown(INSTRUCTIONS)

    with gr.Tabs():
        # ============ TAB 1: UPLOAD ============================================
        with gr.Tab("📤 Upload an image"):
            with gr.Row():
                with gr.Column(scale=1):
                    file_in = gr.File(label="Sentinel-2 image (.tif / .tiff / .png / .jpg)",
                                      file_types=[".tif", ".tiff", ".png", ".jpg", ".jpeg"])
                    area = gr.Radio(list(MODELS.keys()),
                                    value=list(MODELS.keys())[0],
                                    label="Choose model checkpoint set (water type)")
                    with gr.Row():
                        sample_btn = gr.Button("📂 Load sample tile", variant="secondary")
                        run_btn = gr.Button("🚀 Run analysis", variant="primary")
                    color_info = gr.Markdown("")
                with gr.Column(scale=1):
                    gr.Markdown("### Quick checklist\n"
                                "- ⏱ Inference: ~1 s on the VPS GPU.\n"
                                "- 🎯 Best results on tiles from **MagicBathyNet** (matching radiometric scale).\n"
                                "- 🌊 Pick the area that best matches your input's water type.\n"
                                "- 🔍 Pixel-click is enabled on the PhysSDB depth map.")

        # ============ TAB 2: ACQUISITION =======================================
        with gr.Tab("🌍 Acquire from Sentinel-2 S3"):
            gr.Markdown(
                "### Three-step flow:\n"
                "**1.** Draw a rectangle on the map over your **coastal area of interest** "
                "(use the ▭ tool in the top-left of the map).  \n"
                "**2.** Click **🔎 Fetch best Sentinel-2 image** — we search the `sentinel-cogs` AWS S3 bucket "
                "for the lowest-cloud Sentinel-2 L2A scene in your date window and download a chip covering your bbox.  \n"
                "**3.** Inspect the preview, then click **🚀 Send to analysis** to run the baseline U-Net and PhysSDB models on it.\n\n"
                "⚠️ Models were trained on MagicBathyNet (Mediterranean / Baltic). Other water types are "
                "out-of-distribution; PhysSDB's residual map indicates where the model can be trusted."
            )
            with gr.Row():
                with gr.Column(scale=2):
                    gr.HTML(LEAFLET_MAP_HTML)
                with gr.Column(scale=1):
                    with gr.Row():
                        lat_min_in = gr.Number(value=None, label="South lat",  elem_id="sdb_lat_min", precision=5)
                        lat_max_in = gr.Number(value=None, label="North lat",  elem_id="sdb_lat_max", precision=5)
                    with gr.Row():
                        lon_min_in = gr.Number(value=None, label="West lon",   elem_id="sdb_lon_min", precision=5)
                        lon_max_in = gr.Number(value=None, label="East lon",   elem_id="sdb_lon_max", precision=5)
                    days_in = gr.Slider(30, 730, value=365, step=30,
                                        label="Search window (days back from today)")
                    cloud_in = gr.Slider(0.5, 80.0, value=10.0, step=0.5,
                                         label="Max cloud cover (%)")
                    area2 = gr.Radio(list(MODELS.keys()),
                                     value=list(MODELS.keys())[0],
                                     label="Model checkpoint set (water type)")
                    with gr.Row():
                        fetch_btn = gr.Button("🔎 Fetch best Sentinel-2 image", variant="secondary")
                        analyse_btn = gr.Button("🚀 Send to analysis", variant="primary", interactive=False)
                    acq_preview = gr.Image(label="Preview of fetched scene (RGB chip)",
                                           interactive=False)
                    acq_meta = gr.Markdown("*No image fetched yet.*")
            acq_log = gr.Markdown("")
            # Hidden state — stores the fetched chip + scene metadata for the analyse step
            acq_chip_state = gr.State(value=None)

    gr.Markdown("## Inputs and depth predictions")
    with gr.Row():
        with gr.Column():
            in_img = gr.Image(label="Input RGB (preprocessed to 256×256)", interactive=False)
        with gr.Column():
            base_img = gr.Image(label="Baseline U-Net — depth (m)", interactive=False)
        with gr.Column():
            phys_img = gr.Image(label="PhysSDB — depth (m)  [click any pixel]", interactive=False)
        with gr.Column():
            sigma_img = gr.Image(label="PhysSDB — per-pixel σ (m)", interactive=False)

    gr.Markdown("## Physics reconstruction (PhysSDB only)")
    with gr.Row():
        with gr.Column():
            rgb_hat_img = gr.Image(label="PhysSDB reconstructed RGB  (R̂ from Lee-1998)", interactive=False)
        with gr.Column():
            res_img = gr.Image(label="Reconstruction residual  ‖R̂ − R_obs‖  (low = physics fits)",
                               interactive=False)
        with gr.Column():
            gr.Markdown("**How to read this:**\n\n"
                        "- The two left columns should look visually similar where the optical model fits.\n"
                        "- **Bright regions in the residual map** are where Lee-1998 cannot explain the observation "
                        "(typically turbid/Case-2 water, sun glint, or out-of-distribution scenes).\n"
                        "- In bright-residual areas, PhysSDB's depth is less trustworthy — usually accompanied by "
                        "larger predicted σ.")

    gr.Markdown("## Comparison vs measured bathymetry  "
                "*(NOAA NGDC DEM Global Mosaic — composite of multibeam/BAG surveys → ETOPO fallback. "
                "Resolution varies from ~3 m where surveys exist to ~1.85 km in remote ocean. "
                "Land/no-data pixels render white.)*")
    with gr.Row():
        with gr.Column():
            gt_img = gr.Image(label="Measured depth (NOAA reference)", interactive=False)
        with gr.Column():
            err_b_img = gr.Image(label="|Baseline − measured|  (per-pixel abs error)", interactive=False)
        with gr.Column():
            err_p_img = gr.Image(label="|PhysSDB − measured|  (per-pixel abs error)", interactive=False)

    gr.Markdown("## Comparison table")
    table = gr.Dataframe(headers=["Metric", "Baseline U-Net", "PhysSDB"],
                        datatype=["str", "str", "str"], interactive=False)

    gr.Markdown("## Per-pixel inspection (PhysSDB)")
    gr.Markdown("Click a pixel on the **PhysSDB depth** map above to populate this panel.")
    pixel_md = gr.Markdown("*No pixel selected yet.*")

    gr.Markdown("## Downloads")
    download = gr.Files(label="Per-run outputs (`.npz` with all arrays)")

    state = gr.State(value=None)

    sample_btn.click(load_sample, inputs=[area], outputs=[file_in])
    run_btn.click(
        analyse,
        inputs=[file_in, area],
        outputs=[in_img, base_img, phys_img, sigma_img,
                 rgb_hat_img, res_img,
                 gt_img, err_b_img, err_p_img,
                 table, download, state, color_info],
    )
    fetch_btn.click(
        fetch_preview,
        inputs=[lat_min_in, lat_max_in, lon_min_in, lon_max_in, days_in, cloud_in],
        outputs=[acq_preview, acq_meta, acq_chip_state, acq_log, analyse_btn],
    )
    analyse_btn.click(
        analyse_preview,
        inputs=[acq_chip_state, area2],
        outputs=[in_img, base_img, phys_img, sigma_img,
                 rgb_hat_img, res_img,
                 gt_img, err_b_img, err_p_img,
                 table, download, state, color_info],
    )
    phys_img.select(on_pixel_click, inputs=[state], outputs=[pixel_md])

    # Initialize the Leaflet map after the page loads (Gradio 6 strips inline
    # <script> from gr.HTML, so we inject via the `js=` parameter here).
    demo.load(fn=lambda: None, inputs=None, outputs=None, js=LEAFLET_INIT_JS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--share", action="store_true")
    args = ap.parse_args()
    demo.queue(max_size=8).launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        show_error=True,
        favicon_path=None,
        theme="soft",
        allowed_paths=[str(DATA), str(RUNS), str(ROOT)],
    )


if __name__ == "__main__":
    main()
