"""
Cross-site evaluation: load a model trained on one MagicBathyNet area, run
inference on the other area's test set, report RMSE/MAE/R²/bias (+ ECE/cov95 for PhysSDB).

Works for the baseline U-Net, the Stumpf-augmented U-Net, and PhysSDB.

Usage:
   python cross_site.py --ckpt runs/baseline_an/model_best.pt \
                        --model unet --in-channels 3 --stumpf 0 \
                        --src-area agia_napa --dst-area puck_lagoon \
                        --out runs/cross_baseline_an2pl
"""
import argparse, json, os, sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_sdb import (UNet_bathy, load_norm_param, stumpf_channels)
from train_phys import PhysSDB, evaluate_phys


@torch.no_grad()
def eval_unet(net, test_ids, img_tmpl, depth_tmpl, norm_param,
              norm_depth, device, stumpf, crop=256):
    from skimage import io
    from scipy.ndimage import zoom
    net.eval()
    all_pred, all_gt = [], []
    per_tile = {}
    for tid in test_ids:
        img = io.imread(img_tmpl.format(tid)).astype(np.float32)
        if img.ndim == 2: img = img[..., None]
        img = np.transpose(img, (2, 0, 1))[:3]
        pmin = norm_param[0][:img.shape[0], None, None]
        pmax = norm_param[1][:img.shape[0], None, None]
        rng = np.where((pmax - pmin) > 1e-6, pmax - pmin, 1.0)
        img = np.clip((img - pmin) / rng, 0, 1)
        depth = io.imread(depth_tmpl.format(tid)).astype(np.float32) / float(norm_depth)
        ratio = crop / max(img.shape[1], img.shape[2])
        img_z = zoom(img, (1, ratio, ratio), order=1)
        depth_z = zoom(depth, (ratio, ratio), order=1)
        img_z = img_z[:, :crop, :crop]; depth_z = depth_z[:crop, :crop]
        if img_z.shape[1] < crop or img_z.shape[2] < crop:
            ph = crop - img_z.shape[1]; pw = crop - img_z.shape[2]
            img_z = np.pad(img_z, ((0,0),(0,ph),(0,pw)), mode="reflect")
            depth_z = np.pad(depth_z, ((0,ph),(0,pw)), mode="reflect")
        if stumpf:
            img_in = np.concatenate([img_z, stumpf_channels(img_z)], axis=0)
        else:
            img_in = img_z
        x = torch.from_numpy(img_in).unsqueeze(0).float().to(device)
        pred = net(x).cpu().numpy().squeeze()
        img_mask = (img_z.sum(axis=0) != 0).astype(np.float32)
        gt_mask = (depth_z != 0).astype(np.float32)
        mask = (img_mask * gt_mask) > 0.5
        denorm = float(-norm_depth)
        pred_m = pred[mask] * denorm
        gt_m   = depth_z[mask] * denorm
        if pred_m.size:
            per_tile[tid] = dict(
                rmse=float(np.sqrt(np.mean((pred_m - gt_m) ** 2))),
                mae=float(np.mean(np.abs(pred_m - gt_m))),
                bias=float(np.mean(pred_m - gt_m)),
                n=int(pred_m.size),
            )
        all_pred.append(pred_m); all_gt.append(gt_m)
    pred = np.concatenate(all_pred); gt = np.concatenate(all_gt)
    diff = pred - gt
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    mae = float(np.mean(np.abs(diff)))
    bias = float(np.mean(diff))
    ss_res = float(np.sum((gt - pred) ** 2))
    ss_tot = float(np.sum((gt - gt.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return dict(rmse=rmse, mae=mae, bias=bias, r2=r2, per_tile=per_tile,
                pred_all=pred, gt_all=gt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--model", choices=["unet", "phys"], required=True)
    ap.add_argument("--in-channels", type=int, default=3)
    ap.add_argument("--stumpf", type=int, default=0)
    ap.add_argument("--root", default="/home/novarch2/workspace/data/magicbathynet")
    ap.add_argument("--src-area", choices=["agia_napa", "puck_lagoon"], required=True)
    ap.add_argument("--dst-area", choices=["agia_napa", "puck_lagoon"], required=True)
    ap.add_argument("--modality", default="s2")
    ap.add_argument("--crop", type=int, default=256)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # build matching architecture, then load state dict from src
    if args.model == "unet":
        net = UNet_bathy(in_channels=args.in_channels, out_channels=1).to(device)
        net.load_state_dict(torch.load(args.ckpt, map_location=device))
    else:
        # PhysSDB embeds norm_depth_abs as buffer; we restore whatever was trained
        src_norm = 30.443 if args.src_area == "agia_napa" else 11.0
        net = PhysSDB(in_channels=args.in_channels, norm_depth_abs=src_norm).to(device)
        net.load_state_dict(torch.load(args.ckpt, map_location=device))

    # destination data
    droot = Path(args.root) / args.dst_area
    img_tmpl = str(droot / "img" / args.modality / "img_{}.tif")
    depth_tmpl = str(droot / "depth" / args.modality / "depth_{}.tif")
    suffix = "an" if args.dst_area == "agia_napa" else "pl"
    dst_norm_param = load_norm_param(droot / f"norm_param_{args.modality}_{suffix}")
    dst_norm_depth = -30.443 if args.dst_area == "agia_napa" else -11.0
    test_ids = ['411','387','410','398','370','369','397']
    test_ids = [i for i in test_ids if os.path.exists(img_tmpl.format(i))
                                    and os.path.exists(depth_tmpl.format(i))]
    print(f"[info] cross-site: {args.src_area} → {args.dst_area}  ({len(test_ids)} tiles)")

    if args.model == "unet":
        evm = eval_unet(net, test_ids, img_tmpl, depth_tmpl, dst_norm_param,
                        dst_norm_depth, device, stumpf=bool(args.stumpf),
                        crop=args.crop)
    else:
        evm = evaluate_phys(net, test_ids, img_tmpl, depth_tmpl, dst_norm_param,
                            dst_norm_depth, device, crop=args.crop)

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    keep = {k: v for k, v in evm.items()
            if k not in ("first_pred", "first_gt", "first_sigma",
                         "pred_all", "gt_all", "sigma_all")}
    (out / "metrics.json").write_text(json.dumps(
        dict(args=vars(args), final=keep), indent=2))
    np.savez(out / "test_predictions.npz",
             pred=evm["pred_all"], gt=evm["gt_all"],
             **({"sigma": evm["sigma_all"]} if "sigma_all" in evm else {}))
    print(f"[done] cross  rmse={evm['rmse']:.3f}m  mae={evm['mae']:.3f}m  "
          f"r2={evm['r2']:.3f}" +
          (f"  ECE={evm.get('ece', float('nan')):.3f}  "
           f"cov95={evm.get('pi_coverage_95', float('nan')):.3f}"
           if args.model == "phys" else ""))


if __name__ == "__main__":
    main()
