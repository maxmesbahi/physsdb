"""
Sentinel-2 Bathymetry training script for MagicBathyNet.

Baseline      : 3-channel RGB U-Net (reproduces MagicBathyNet bathymetry baseline).
Stumpf variant: 5-channel input = [R,G,B, log(n*B+eps)/log(n*G+eps), log(n*B+eps)/log(n*R+eps)]
                The two extra channels are physics-informed Stumpf log-ratio features
                that correlate monotonically with depth in clear shallow water.

Usage:
  python train_sdb.py --area agia_napa --modality s2 --stumpf 0 --epochs 30 --out runs/baseline_an
  python train_sdb.py --area agia_napa --modality s2 --stumpf 1 --epochs 30 --out runs/stumpf_an

Outputs (under --out):
  metrics.json           per-epoch + final train/val metrics
  test_predictions.npz   per-test-tile prediction + GT arrays (denormalized to meters)
  scatter.png            pred vs GT scatter
  pred_map_<id>.png      first test tile prediction overlay
  model_final.pt         final weights
"""
import argparse, json, os, random, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# -------------------- U-Net (vendored from MagicBathyNet, parameterized) --------------------
class DoubleConv(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1), nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.net(x)

class Down(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.net = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_c, out_c))
    def forward(self, x): return self.net(x)

class Up(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_c, out_c, 2, stride=2)
        self.conv = DoubleConv(in_c, out_c)
    def forward(self, x1, x2):
        x1 = self.up(x1)
        dy, dx = x2.size(2) - x1.size(2), x2.size(3) - x1.size(3)
        x1 = F.pad(x1, [dx // 2, dx - dx // 2, dy // 2, dy - dy // 2])
        return self.conv(torch.cat([x2, x1], dim=1))

class UNet_bathy(nn.Module):
    def __init__(self, in_channels, out_channels=1):
        super().__init__()
        self.enc1 = DoubleConv(in_channels, 32)
        self.enc2 = Down(32, 64)
        self.enc3 = Down(64, 128)
        self.enc4 = Down(128, 256)
        self.dec3 = Up(256, 128)
        self.dec2 = Up(128, 64)
        self.dec1 = Up(64, 32)
        self.head = nn.Conv2d(32, out_channels, 1)
    def forward(self, x):
        x1 = self.enc1(x); x2 = self.enc2(x1); x3 = self.enc3(x2); x4 = self.enc4(x3)
        x = self.dec3(x4, x3); x = self.dec2(x, x2); x = self.dec1(x, x1)
        return self.head(x)

# -------------------- helpers --------------------
def load_norm_param(path):
    """Load norm_param: prefer .npy, fallback to .txt (whitespace-separated 2xN)."""
    if path.with_suffix(".npy").exists():
        return np.load(path.with_suffix(".npy"))
    if path.with_suffix(".txt").exists():
        arr = np.loadtxt(path.with_suffix(".txt"))
        if arr.ndim == 1:  # single row → assume (min;max) interleaved? coerce to (2,N)
            arr = arr.reshape(2, -1)
        return arr
    # try without extension change
    if path.exists():
        try:
            return np.load(path)
        except Exception:
            return np.loadtxt(path)
    raise FileNotFoundError(f"No norm_param found at {path}(.npy|.txt)")

def read_split(path):
    """Split file: read tile IDs (one per line, may have suffix or extension)."""
    ids = []
    for ln in Path(path).read_text().splitlines():
        ln = ln.strip()
        if not ln: continue
        # extract numeric/string id; strip extensions
        ln = ln.replace(".tif", "").replace("img_", "").replace("depth_", "")
        ids.append(ln)
    return ids

def stumpf_channels(img, n=1000.0, eps=1e-6):
    """img: float32 (C=3,H,W) in [0,1]. Returns (2,H,W) Stumpf log-ratio channels.
    Channel 0: log(n*B+eps)/log(n*G+eps); Channel 1: log(n*B+eps)/log(n*R+eps).
    Assumes RGB band order (channel 0=R, 1=G, 2=B)."""
    R, G, B = img[0], img[1], img[2]
    nB = np.log(n * np.clip(B, 0, 1) + eps)
    nG = np.log(n * np.clip(G, 0, 1) + eps)
    nR = np.log(n * np.clip(R, 0, 1) + eps)
    s_bg = nB / np.where(np.abs(nG) < eps, eps, nG)
    s_br = nB / np.where(np.abs(nR) < eps, eps, nR)
    return np.stack([s_bg.astype(np.float32), s_br.astype(np.float32)], axis=0)

# -------------------- dataset --------------------
class SDBDataset(Dataset):
    def __init__(self, ids, img_tmpl, depth_tmpl, norm_param, norm_depth,
                 window_size, samples_per_epoch=4000, augment=True, stumpf=False):
        self.ids = ids
        self.img_tmpl = img_tmpl
        self.depth_tmpl = depth_tmpl
        self.norm_param = norm_param.astype(np.float32)
        self.norm_depth = float(norm_depth)
        self.W = window_size
        self.N = samples_per_epoch
        self.augment = augment
        self.stumpf = stumpf
        self._cache = {}

    def __len__(self): return self.N

    def _load(self, tid):
        if tid in self._cache: return self._cache[tid]
        from skimage import io
        ip = self.img_tmpl.format(tid); dp = self.depth_tmpl.format(tid)
        img = io.imread(ip).astype(np.float32)         # H W C
        depth = io.imread(dp).astype(np.float32)        # H W
        if img.ndim == 2: img = img[..., None]
        # transpose to C H W
        img = np.transpose(img, (2, 0, 1))
        # take first 3 channels (some tifs may have alpha)
        img = img[:3]
        # normalize per channel
        pmin = self.norm_param[0][:img.shape[0], None, None]
        pmax = self.norm_param[1][:img.shape[0], None, None]
        rng = np.where((pmax - pmin) > 1e-6, (pmax - pmin), 1.0)
        img = (img - pmin) / rng
        img = np.clip(img, 0.0, 1.0)
        depth = depth / self.norm_depth   # negative depths → positive [0,1]
        self._cache[tid] = (img, depth)
        return img, depth

    def __getitem__(self, idx):
        tid = random.choice(self.ids)
        img, depth = self._load(tid)
        C, H, W = img.shape
        w, h = self.W
        if H < h or W < w:
            # patch the whole image (pad if needed)
            pad_h = max(0, h - H); pad_w = max(0, w - W)
            if pad_h or pad_w:
                img = np.pad(img, ((0,0),(0,pad_h),(0,pad_w)), mode='reflect')
                depth = np.pad(depth, ((0,pad_h),(0,pad_w)), mode='reflect')
            x1, y1 = 0, 0
        else:
            x1 = random.randint(0, H - h); y1 = random.randint(0, W - w)
        img_p = img[:, x1:x1+h, y1:y1+w].copy()
        depth_p = depth[x1:x1+h, y1:y1+w].copy()
        if self.augment:
            if random.random() < 0.5:
                img_p = img_p[:, ::-1, :].copy(); depth_p = depth_p[::-1, :].copy()
            if random.random() < 0.5:
                img_p = img_p[:, :, ::-1].copy(); depth_p = depth_p[:, ::-1].copy()
        if self.stumpf:
            extra = stumpf_channels(img_p)
            img_p = np.concatenate([img_p, extra], axis=0)
        return torch.from_numpy(img_p), torch.from_numpy(depth_p)

# -------------------- loss --------------------
class MaskedRMSE(nn.Module):
    def forward(self, output, depth, mask):
        diff = (output.squeeze(1) - depth) ** 2
        m = mask.float()
        n = m.sum().clamp_min(1.0)
        return torch.sqrt((diff * m).sum() / n)

# -------------------- evaluate --------------------
@torch.no_grad()
def evaluate(net, test_ids, img_tmpl, depth_tmpl, norm_param, norm_depth,
             device, stumpf, crop=256):
    from skimage import io
    from scipy.ndimage import zoom
    net.eval()
    all_pred_m, all_gt_m = [], []
    per_tile = {}
    first_pred = None; first_gt = None; first_id = None
    for tid in test_ids:
        ip = img_tmpl.format(tid); dp = depth_tmpl.format(tid)
        img = io.imread(ip).astype(np.float32)
        if img.ndim == 2: img = img[..., None]
        img = np.transpose(img, (2, 0, 1))[:3]
        pmin = norm_param[0][:img.shape[0], None, None]
        pmax = norm_param[1][:img.shape[0], None, None]
        rng = np.where((pmax - pmin) > 1e-6, (pmax - pmin), 1.0)
        img = np.clip((img - pmin) / rng, 0.0, 1.0)
        depth = io.imread(dp).astype(np.float32) / float(norm_depth)
        # upscale to (crop x crop) for the network
        C, H, W = img.shape
        ratio = crop / max(H, W)
        img_z = zoom(img, (1, ratio, ratio), order=1)
        depth_z = zoom(depth, (ratio, ratio), order=1)
        # ensure exactly crop x crop
        img_z = img_z[:, :crop, :crop]
        depth_z = depth_z[:crop, :crop]
        if img_z.shape[1] < crop or img_z.shape[2] < crop:
            ph = crop - img_z.shape[1]; pw = crop - img_z.shape[2]
            img_z = np.pad(img_z, ((0,0),(0,ph),(0,pw)), mode='reflect')
            depth_z = np.pad(depth_z, ((0,ph),(0,pw)), mode='reflect')
        if stumpf:
            extra = stumpf_channels(img_z)
            img_in = np.concatenate([img_z, extra], axis=0)
        else:
            img_in = img_z
        x = torch.from_numpy(img_in).unsqueeze(0).float().to(device)
        pred = net(x).cpu().numpy().squeeze()
        # mask: image not zero AND depth not zero
        img_mask = (img_z.sum(axis=0) != 0).astype(np.float32)
        gt_mask = (depth_z != 0).astype(np.float32)
        mask = (img_mask * gt_mask) > 0.5
        # denormalize to meters (positive depths)
        pred_m = pred[mask] * float(-norm_depth)
        gt_m = depth_z[mask] * float(-norm_depth)
        all_pred_m.append(pred_m); all_gt_m.append(gt_m)
        if pred_m.size:
            per_tile[tid] = dict(
                rmse=float(np.sqrt(np.mean((pred_m - gt_m) ** 2))),
                mae=float(np.mean(np.abs(pred_m - gt_m))),
                bias=float(np.mean(pred_m - gt_m)),
                n=int(pred_m.size),
            )
        if first_pred is None and pred_m.size:
            first_pred = pred * float(-norm_depth)
            first_gt = depth_z * float(-norm_depth)
            first_id = tid
    pred = np.concatenate(all_pred_m); gt = np.concatenate(all_gt_m)
    diff = pred - gt
    rmse = float(np.sqrt(np.mean(diff**2)))
    mae = float(np.mean(np.abs(diff)))
    bias = float(np.mean(diff))
    ss_res = float(np.sum((gt - pred) ** 2))
    ss_tot = float(np.sum((gt - gt.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    # per-depth-bin
    bins = [(0,2),(2,5),(5,10),(10,20),(20,35)]
    bin_metrics = {}
    for lo, hi in bins:
        sel = (gt >= lo) & (gt < hi)
        if sel.sum() > 0:
            d = pred[sel] - gt[sel]
            bin_metrics[f"{lo}-{hi}m"] = dict(
                n=int(sel.sum()), rmse=float(np.sqrt(np.mean(d**2))),
                mae=float(np.mean(np.abs(d))), bias=float(np.mean(d)),
            )
    return dict(rmse=rmse, mae=mae, bias=bias, r2=r2,
                per_tile=per_tile, per_bin=bin_metrics,
                first_pred=first_pred, first_gt=first_gt, first_id=first_id,
                pred_all=pred, gt_all=gt)

# -------------------- train loop --------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/home/novarch2/workspace/data/magicbathynet",
                    help="path to extracted magicbathynet root")
    ap.add_argument("--area", choices=["agia_napa", "puck_lagoon"], default="agia_napa")
    ap.add_argument("--modality", default="s2")
    ap.add_argument("--stumpf", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--samples-per-epoch", type=int, default=4000)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--crop", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    root = Path(args.root) / args.area
    img_tmpl = str(root / "img" / args.modality / "img_{}.tif")
    depth_tmpl = str(root / "depth" / args.modality / "depth_{}.tif")
    suffix = "an" if args.area == "agia_napa" else "pl"
    norm_param_path = root / f"norm_param_{args.modality}_{suffix}"
    norm_param = load_norm_param(norm_param_path)
    norm_depth = -30.443 if args.area == "agia_napa" else -11.0

    # splits — fall back to the IDs hardcoded in the upstream notebook if missing.
    split_file = root / f"{args.modality}_split_bathymetry.txt"
    if split_file.exists():
        all_ids = read_split(split_file)
    else:
        # IDs from the original MagicBathyNet notebook (Agia Napa). Same set used at Puck if file missing.
        all_ids = ['409','418','350','399','361','430','380','359','371','377','379','360',
                   '368','419','389','420','401','408','352','388','362','421','412','351',
                   '349','390','400','378','411','387','410','398','370','369','397']
    train_ids = ['409','418','350','399','361','430','380','359','371','377','379','360',
                 '368','419','389','420','401','408','352','388','362','421','412','351',
                 '349','390','400','378']
    test_ids = ['411','387','410','398','370','369','397']
    # keep only those actually on disk
    train_ids = [i for i in train_ids if os.path.exists(img_tmpl.format(i)) and os.path.exists(depth_tmpl.format(i))]
    test_ids = [i for i in test_ids if os.path.exists(img_tmpl.format(i)) and os.path.exists(depth_tmpl.format(i))]
    print(f"[info] area={args.area} modality={args.modality} stumpf={args.stumpf}")
    print(f"[info] train_ids ({len(train_ids)}): {train_ids[:5]}…")
    print(f"[info] test_ids  ({len(test_ids)}): {test_ids}")

    # window size: small (18) for S2 means we let it patch the whole image.
    if args.modality == "s2": WIN = (18, 18)
    elif args.modality == "spot6": WIN = (30, 30)
    else: WIN = (256, 256)

    in_ch = 3 + (2 if args.stumpf else 0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[info] device={device}, input channels={in_ch}")

    train_ds = SDBDataset(train_ids, img_tmpl, depth_tmpl, norm_param, norm_depth,
                          window_size=WIN, samples_per_epoch=args.samples_per_epoch,
                          augment=True, stumpf=bool(args.stumpf))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=2, pin_memory=True)
    net = UNet_bathy(in_channels=in_ch).to(device)
    optim_ = optim.Adam(net.parameters(), lr=args.lr)
    sched = optim.lr_scheduler.MultiStepLR(optim_, [args.epochs - 5], gamma=0.1)
    loss_fn = MaskedRMSE()

    metrics_log = []
    best_rmse = float("inf")
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        net.train(); ep_loss = 0.0; n_batches = 0
        for x, y in train_loader:
            x = x.float().to(device, non_blocking=True)
            y = y.float().to(device, non_blocking=True)
            # upsample patches to crop x crop
            x_in = F.interpolate(x, size=(args.crop, args.crop), mode='bilinear', align_corners=False)
            y_in = F.interpolate(y.unsqueeze(1), size=(args.crop, args.crop), mode='nearest').squeeze(1)
            mask = (y_in != 0) & (x_in[:, :3].sum(dim=1) != 0)
            if mask.sum() == 0: continue
            out_ = net(x_in)
            loss = loss_fn(out_, y_in, mask)
            optim_.zero_grad(); loss.backward(); optim_.step()
            ep_loss += loss.item(); n_batches += 1
        sched.step()
        mean_loss = ep_loss / max(1, n_batches)
        # quick eval
        evm = evaluate(net, test_ids, img_tmpl, depth_tmpl, norm_param, norm_depth,
                       device, bool(args.stumpf), crop=args.crop)
        ent = dict(epoch=ep, train_loss=mean_loss, test_rmse=evm["rmse"],
                   test_mae=evm["mae"], test_bias=evm["bias"], test_r2=evm["r2"])
        metrics_log.append(ent)
        elapsed = time.time() - t0
        print(f"[ep {ep:02d}] loss={mean_loss:.4f}  test_rmse={evm['rmse']:.3f}m  "
              f"mae={evm['mae']:.3f}m  bias={evm['bias']:+.3f}m  r2={evm['r2']:.3f}  "
              f"({elapsed/60:.1f} min)")
        if evm["rmse"] < best_rmse:
            best_rmse = evm["rmse"]
            torch.save(net.state_dict(), out / "model_best.pt")

    torch.save(net.state_dict(), out / "model_final.pt")
    # final eval with figures
    final = evaluate(net, test_ids, img_tmpl, depth_tmpl, norm_param, norm_depth,
                     device, bool(args.stumpf), crop=args.crop)
    # save metrics
    serializable_final = {k: v for k, v in final.items()
                          if k not in ("first_pred", "first_gt", "pred_all", "gt_all")}
    out_json = dict(
        args=vars(args), epochs_log=metrics_log, final=serializable_final,
        best_rmse=best_rmse, wall_min=(time.time()-t0)/60.0,
    )
    (out / "metrics.json").write_text(json.dumps(out_json, indent=2))
    np.savez(out / "test_predictions.npz",
             pred=final["pred_all"], gt=final["gt_all"])
    # figures
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(5,5))
        ax.scatter(final["gt_all"], final["pred_all"], s=2, alpha=0.3)
        lo = float(min(final["gt_all"].min(), final["pred_all"].min()))
        hi = float(max(final["gt_all"].max(), final["pred_all"].max()))
        ax.plot([lo, hi], [lo, hi], 'r--', lw=1)
        ax.set_xlabel("Reference depth (m)"); ax.set_ylabel("Predicted depth (m)")
        ax.set_title(f"{args.area} {args.modality} stumpf={args.stumpf}\n"
                     f"RMSE={final['rmse']:.2f} m   R²={final['r2']:.3f}   bias={final['bias']:+.2f}")
        fig.tight_layout(); fig.savefig(out / "scatter.png", dpi=130); plt.close(fig)
        if final["first_pred"] is not None:
            fig, axes = plt.subplots(1, 2, figsize=(8,4))
            vmax = max(final["first_pred"].max(), final["first_gt"].max())
            axes[0].imshow(final["first_gt"], vmin=0, vmax=vmax, cmap="viridis_r")
            axes[0].set_title(f"Reference (tile {final['first_id']})")
            axes[1].imshow(final["first_pred"], vmin=0, vmax=vmax, cmap="viridis_r")
            axes[1].set_title("Prediction")
            for a in axes: a.axis("off")
            fig.tight_layout(); fig.savefig(out / f"pred_map_{final['first_id']}.png", dpi=130); plt.close(fig)
    except Exception as e:
        print("[warn] figure save failed:", e)

    print(f"[done] best_rmse={best_rmse:.3f}m  final_rmse={final['rmse']:.3f}m   "
          f"wrote {out / 'metrics.json'}")

if __name__ == "__main__":
    main()
