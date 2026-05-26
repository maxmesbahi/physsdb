"""
PhysSDB: Physics-Constrained Uncertainty-Aware Sentinel-2 Bathymetry.

Single U-Net trunk → 7 output channels split as:
  d_norm        (1)  sigmoid → normalized depth in [0,1]
  a, b_b        (2)  sigmoid → absorption and backscatter (scaled)
  rho_R,G,B     (3)  sigmoid → per-band bottom reflectance (scaled to [0,0.3])
  log_sigma     (1)  unbounded → heteroscedastic uncertainty

A differentiable Lee-1998-style forward model reconstructs the per-band
reflectance from {d, a, b_b, rho, R_inf}, and reflectance reconstruction error
becomes a self-supervised physics loss requiring no extra labels.

Loss:
   L = L_data_NLL + λ_phys · L_physics

   L_data_NLL = mean over masked pixels of
                  0.5 * (d̂ - d)² / σ² + log σ
   L_physics  = mean over masked pixels and bands of (R̂ - R_obs)²

Usage:
   python train_phys.py --area agia_napa --epochs 40 --lambda-phys 1.0 \
                        --out runs/physsdb_an
"""
import argparse, json, os, random, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

# reuse from baseline script (must sit next to this file)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_sdb import (UNet_bathy, SDBDataset, load_norm_param, read_split, stumpf_channels)


# ----------------------------------------------------------------------
# PhysSDB model
# ----------------------------------------------------------------------
class PhysSDB(nn.Module):
    """Multi-head U-Net with differentiable Lee-1998 forward model."""

    def __init__(self, in_channels=3, norm_depth_abs=30.443):
        super().__init__()
        self.trunk = UNet_bathy(in_channels=in_channels, out_channels=7)
        # learnable per-band deep-water reflectance (in log space, init ~0.01)
        self.log_R_inf = nn.Parameter(torch.log(torch.tensor([0.005, 0.010, 0.020])))
        # Training-time depth normalizer (|min_depth|, meters). Saved with state_dict.
        self.register_buffer("norm_depth_abs", torch.tensor(float(norm_depth_abs)))
        # First 3 input channels are always assumed to be (R, G, B) reflectance proxies.
        # If in_channels > 3, the extra channels are ignored by the physics module.

    def forward(self, x):
        raw = self.trunk(x)                                       # [B,7,H,W]
        d_norm     = torch.sigmoid(raw[:, 0:1])                   # [0,1]
        a          = torch.sigmoid(raw[:, 1:2]) * 1.0             # m^-1, normalized scale
        b_b        = torch.sigmoid(raw[:, 2:3]) * 0.5
        rho        = torch.sigmoid(raw[:, 3:6]) * 0.30            # bottom reflectance per band
        log_sigma  = raw[:, 6:7].clamp(min=-7.0, max=3.0)         # numerical stability

        # Lee-1998-style forward model (simplified, per band)
        # depth in meters:
        d_m = d_norm * self.norm_depth_abs
        K = 2.0 * (a + b_b)                                       # [B,1,H,W]
        R_inf = torch.exp(self.log_R_inf).view(1, 3, 1, 1)        # [1,3,1,1]
        atten = torch.exp(-K * d_m)                               # [B,1,H,W]
        R_hat = rho * atten + R_inf * (1.0 - atten)               # [B,3,H,W]

        return dict(d=d_norm, a=a, b_b=b_b, rho=rho,
                    log_sigma=log_sigma, R_hat=R_hat)


# ----------------------------------------------------------------------
# Losses
# ----------------------------------------------------------------------
def hetero_nll_and_physics(out, x_obs_rgb, d_target_norm, mask, lambda_phys=1.0):
    """
    out          : dict from PhysSDB.forward
    x_obs_rgb    : [B,3,H,W] observed normalized R,G,B (in [0,1])
    d_target_norm: [B,H,W] normalized depth target (in [0,1])
    mask         : [B,H,W] bool tensor (where loss applies)
    """
    d_hat = out["d"].squeeze(1)                                   # [B,H,W]
    log_sigma = out["log_sigma"].squeeze(1)                       # [B,H,W]
    sigma2 = torch.exp(2.0 * log_sigma)
    nll = 0.5 * (d_hat - d_target_norm) ** 2 / sigma2 + log_sigma
    m = mask.float()
    n = m.sum().clamp_min(1.0)
    L_data = (nll * m).sum() / n

    R_hat = out["R_hat"]                                          # [B,3,H,W]
    # physics mask: only underwater pixels (have depth) with non-zero input
    img_valid = (x_obs_rgb.sum(dim=1) != 0)
    phys_mask = (mask & img_valid).unsqueeze(1).expand_as(R_hat).float()
    npx = phys_mask.sum().clamp_min(1.0)
    L_phys = ((R_hat - x_obs_rgb) ** 2 * phys_mask).sum() / npx

    return L_data + lambda_phys * L_phys, dict(L_data=L_data.item(),
                                               L_phys=L_phys.item())


# ----------------------------------------------------------------------
# Uncertainty metrics
# ----------------------------------------------------------------------
def expected_calibration_error(errors_m, sigmas_m, n_bins=15):
    """ECE between empirical RMSE and mean predicted sigma, binned by sigma."""
    if len(sigmas_m) < n_bins * 5:
        return float("nan")
    edges = np.quantile(sigmas_m, np.linspace(0, 1, n_bins + 1))
    ece = 0.0
    n_total = len(sigmas_m)
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        sel = (sigmas_m >= lo) & (sigmas_m <= hi if i == n_bins - 1 else sigmas_m < hi)
        k = int(sel.sum())
        if k < 10: continue
        emp_rmse = float(np.sqrt(np.mean(errors_m[sel] ** 2)))
        mean_sig = float(np.mean(sigmas_m[sel]))
        ece += (k / n_total) * abs(emp_rmse - mean_sig)
    return float(ece)


def pi_coverage(errors_m, sigmas_m, z=1.96):
    """Empirical coverage of the z·σ predictive interval (target 0.95 for z=1.96)."""
    if len(sigmas_m) == 0: return float("nan")
    return float(np.mean(np.abs(errors_m) <= z * sigmas_m))


# ----------------------------------------------------------------------
# Evaluation (mirrors train_sdb.evaluate but emits uncertainty)
# ----------------------------------------------------------------------
@torch.no_grad()
def evaluate_phys(net, test_ids, img_tmpl, depth_tmpl, norm_param,
                  norm_depth, device, crop=256):
    from skimage import io
    from scipy.ndimage import zoom
    net.eval()
    all_pred, all_gt, all_sig = [], [], []
    per_tile = {}
    first_pred = first_gt = first_id = first_sig = None

    for tid in test_ids:
        img = io.imread(img_tmpl.format(tid)).astype(np.float32)
        if img.ndim == 2: img = img[..., None]
        img = np.transpose(img, (2, 0, 1))[:3]
        pmin = norm_param[0][:img.shape[0], None, None]
        pmax = norm_param[1][:img.shape[0], None, None]
        rng = np.where((pmax - pmin) > 1e-6, pmax - pmin, 1.0)
        img = np.clip((img - pmin) / rng, 0, 1)
        depth = io.imread(depth_tmpl.format(tid)).astype(np.float32) / float(norm_depth)
        C, H, W = img.shape
        ratio = crop / max(H, W)
        img_z = zoom(img, (1, ratio, ratio), order=1)
        depth_z = zoom(depth, (ratio, ratio), order=1)
        img_z = img_z[:, :crop, :crop]; depth_z = depth_z[:crop, :crop]
        if img_z.shape[1] < crop or img_z.shape[2] < crop:
            ph = crop - img_z.shape[1]; pw = crop - img_z.shape[2]
            img_z = np.pad(img_z, ((0,0),(0,ph),(0,pw)), mode="reflect")
            depth_z = np.pad(depth_z, ((0,ph),(0,pw)), mode="reflect")
        x = torch.from_numpy(img_z).unsqueeze(0).float().to(device)
        out = net(x)
        d_pred = out["d"].squeeze().cpu().numpy()
        sig = torch.exp(out["log_sigma"]).squeeze().cpu().numpy()   # normalized-depth sigma

        img_mask = (img_z.sum(axis=0) != 0).astype(np.float32)
        gt_mask = (depth_z != 0).astype(np.float32)
        mask = (img_mask * gt_mask) > 0.5

        # de-normalize to meters
        denorm = float(-norm_depth)        # negative → meters magnitude
        pred_m = d_pred[mask] * denorm
        gt_m   = depth_z[mask] * denorm
        sig_m  = sig[mask] * denorm
        all_pred.append(pred_m); all_gt.append(gt_m); all_sig.append(sig_m)

        if pred_m.size:
            per_tile[tid] = dict(
                rmse=float(np.sqrt(np.mean((pred_m - gt_m) ** 2))),
                mae=float(np.mean(np.abs(pred_m - gt_m))),
                bias=float(np.mean(pred_m - gt_m)),
                n=int(pred_m.size),
            )
        if first_pred is None and pred_m.size:
            first_pred = d_pred * denorm
            first_gt = depth_z * denorm
            first_sig = sig * denorm
            first_id = tid

    pred = np.concatenate(all_pred); gt = np.concatenate(all_gt); sig = np.concatenate(all_sig)
    diff = pred - gt
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    mae = float(np.mean(np.abs(diff)))
    bias = float(np.mean(diff))
    ss_res = float(np.sum((gt - pred) ** 2))
    ss_tot = float(np.sum((gt - gt.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    ece = expected_calibration_error(np.abs(diff), sig)
    cov95 = pi_coverage(diff, sig)
    mean_sig = float(np.mean(sig))

    # per-depth-bin
    bins = [(0,2),(2,5),(5,10),(10,20),(20,35)]
    per_bin = {}
    for lo, hi in bins:
        sel = (gt >= lo) & (gt < hi)
        if sel.sum() > 0:
            d = pred[sel] - gt[sel]
            per_bin[f"{lo}-{hi}m"] = dict(
                n=int(sel.sum()), rmse=float(np.sqrt(np.mean(d**2))),
                mae=float(np.mean(np.abs(d))), bias=float(np.mean(d)),
            )

    return dict(rmse=rmse, mae=mae, bias=bias, r2=r2,
                ece=ece, pi_coverage_95=cov95, mean_sigma_m=mean_sig,
                per_tile=per_tile, per_bin=per_bin,
                first_pred=first_pred, first_gt=first_gt,
                first_sigma=first_sig, first_id=first_id,
                pred_all=pred, gt_all=gt, sigma_all=sig)


# ----------------------------------------------------------------------
# Train loop
# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/home/novarch2/workspace/data/magicbathynet")
    ap.add_argument("--area", choices=["agia_napa", "puck_lagoon"], default="agia_napa")
    ap.add_argument("--modality", default="s2")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--samples-per-epoch", type=int, default=4000)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--crop", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lambda-phys", type=float, default=1.0)
    ap.add_argument("--lambda-phys-warmup", type=int, default=5,
                    help="ramp lambda_phys linearly from 0 to its final value over N epochs")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    root = Path(args.root) / args.area
    img_tmpl = str(root / "img" / args.modality / "img_{}.tif")
    depth_tmpl = str(root / "depth" / args.modality / "depth_{}.tif")
    suffix = "an" if args.area == "agia_napa" else "pl"
    norm_param = load_norm_param(root / f"norm_param_{args.modality}_{suffix}")
    norm_depth = -30.443 if args.area == "agia_napa" else -11.0

    train_ids = ['409','418','350','399','361','430','380','359','371','377','379','360',
                 '368','419','389','420','401','408','352','388','362','421','412','351',
                 '349','390','400','378']
    test_ids  = ['411','387','410','398','370','369','397']
    train_ids = [i for i in train_ids if os.path.exists(img_tmpl.format(i))
                                      and os.path.exists(depth_tmpl.format(i))]
    test_ids  = [i for i in test_ids  if os.path.exists(img_tmpl.format(i))
                                      and os.path.exists(depth_tmpl.format(i))]
    print(f"[info] area={args.area} train={len(train_ids)} test={len(test_ids)}  "
          f"λ_phys={args.lambda_phys} (warmup {args.lambda_phys_warmup} ep)")

    WIN = (18, 18) if args.modality == "s2" else (30, 30)

    train_ds = SDBDataset(train_ids, img_tmpl, depth_tmpl, norm_param, norm_depth,
                          window_size=WIN, samples_per_epoch=args.samples_per_epoch,
                          augment=True, stumpf=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=2, pin_memory=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = PhysSDB(in_channels=3, norm_depth_abs=abs(norm_depth)).to(device)
    optim_ = optim.Adam(net.parameters(), lr=args.lr)
    sched = optim.lr_scheduler.MultiStepLR(optim_, [max(1, args.epochs - 5)], gamma=0.1)

    metrics_log = []
    best_rmse = float("inf")
    t0 = time.time()

    for ep in range(1, args.epochs + 1):
        # linear warmup of physics loss
        lam = args.lambda_phys * min(1.0, ep / max(1, args.lambda_phys_warmup))
        net.train()
        ep_loss = ep_Ldata = ep_Lphys = 0.0
        nb = 0
        for x, y in train_loader:
            x = x.float().to(device, non_blocking=True)
            y = y.float().to(device, non_blocking=True)
            x_in = F.interpolate(x, size=(args.crop, args.crop),
                                 mode="bilinear", align_corners=False)
            y_in = F.interpolate(y.unsqueeze(1), size=(args.crop, args.crop),
                                 mode="nearest").squeeze(1)
            mask = (y_in != 0) & (x_in[:, :3].sum(dim=1) != 0)
            if mask.sum() == 0: continue
            out_ = net(x_in[:, :3])
            loss, parts = hetero_nll_and_physics(out_, x_in[:, :3], y_in, mask,
                                                 lambda_phys=lam)
            optim_.zero_grad(); loss.backward(); optim_.step()
            ep_loss += loss.item(); ep_Ldata += parts["L_data"]; ep_Lphys += parts["L_phys"]
            nb += 1
        sched.step()
        evm = evaluate_phys(net, test_ids, img_tmpl, depth_tmpl, norm_param,
                            norm_depth, device, crop=args.crop)
        ent = dict(epoch=ep, lam=lam,
                   train_loss=ep_loss / max(1, nb),
                   L_data=ep_Ldata / max(1, nb),
                   L_phys=ep_Lphys / max(1, nb),
                   test_rmse=evm["rmse"], test_mae=evm["mae"],
                   test_bias=evm["bias"], test_r2=evm["r2"],
                   ece=evm["ece"], pi95=evm["pi_coverage_95"],
                   mean_sigma_m=evm["mean_sigma_m"])
        metrics_log.append(ent)
        elapsed = (time.time() - t0) / 60.0
        print(f"[ep {ep:02d}] L={ent['train_loss']:.4f} "
              f"(data={ent['L_data']:.4f} phys={ent['L_phys']:.4f} λ={lam:.2f})  "
              f"rmse={evm['rmse']:.3f}m  mae={evm['mae']:.3f}m  "
              f"r2={evm['r2']:.3f}  ECE={evm['ece']:.3f}  cov95={evm['pi_coverage_95']:.3f}  "
              f"({elapsed:.1f}m)")
        if evm["rmse"] < best_rmse:
            best_rmse = evm["rmse"]
            torch.save(net.state_dict(), out / "model_best.pt")

    torch.save(net.state_dict(), out / "model_final.pt")
    final = evaluate_phys(net, test_ids, img_tmpl, depth_tmpl, norm_param,
                          norm_depth, device, crop=args.crop)
    ser = {k: v for k, v in final.items()
           if k not in ("first_pred", "first_gt", "first_sigma",
                        "pred_all", "gt_all", "sigma_all")}
    (out / "metrics.json").write_text(json.dumps(
        dict(args=vars(args), epochs_log=metrics_log, final=ser,
             best_rmse=best_rmse, wall_min=(time.time() - t0) / 60), indent=2))
    np.savez(out / "test_predictions.npz",
             pred=final["pred_all"], gt=final["gt_all"], sigma=final["sigma_all"])

    # figures
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        # scatter + sigma colour
        fig, ax = plt.subplots(figsize=(5, 5))
        sc = ax.scatter(final["gt_all"], final["pred_all"], s=2, alpha=0.3,
                        c=final["sigma_all"], cmap="viridis")
        lo = float(min(final["gt_all"].min(), final["pred_all"].min()))
        hi = float(max(final["gt_all"].max(), final["pred_all"].max()))
        ax.plot([lo, hi], [lo, hi], "r--", lw=1)
        ax.set_xlabel("Reference depth (m)"); ax.set_ylabel("Predicted depth (m)")
        ax.set_title(f"PhysSDB {args.area}\n"
                     f"RMSE={final['rmse']:.2f}m  R²={final['r2']:.3f}  "
                     f"ECE={final['ece']:.3f}  95%PI={final['pi_coverage_95']:.3f}")
        plt.colorbar(sc, ax=ax, label="σ (m)")
        fig.tight_layout(); fig.savefig(out / "scatter.png", dpi=130); plt.close(fig)

        if final["first_pred"] is not None:
            fig, ax = plt.subplots(1, 3, figsize=(11, 4))
            vmax = max(final["first_pred"].max(), final["first_gt"].max())
            ax[0].imshow(final["first_gt"], vmin=0, vmax=vmax, cmap="viridis_r")
            ax[0].set_title(f"Reference (tile {final['first_id']})")
            ax[1].imshow(final["first_pred"], vmin=0, vmax=vmax, cmap="viridis_r")
            ax[1].set_title("PhysSDB prediction")
            ax[2].imshow(final["first_sigma"], cmap="magma")
            ax[2].set_title("PhysSDB σ (m)")
            for a in ax: a.axis("off")
            fig.tight_layout(); fig.savefig(out / f"pred_map_{final['first_id']}.png",
                                           dpi=130); plt.close(fig)
    except Exception as e:
        print("[warn] figures:", e)

    print(f"[done] best={best_rmse:.3f}m  final={final['rmse']:.3f}m  "
          f"ECE={final['ece']:.3f}  cov95={final['pi_coverage_95']:.3f}")


if __name__ == "__main__":
    main()
