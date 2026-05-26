"""Aggregate per-run metrics into a single markdown comparison table + side-by-side scatters."""
import argparse, json, sys
from pathlib import Path

import numpy as np


def load(run_dir: Path):
    f = run_dir / "metrics.json"
    if not f.exists(): return None
    return json.loads(f.read_text())


def fmt_row(label, m):
    if m is None:
        return f"| {label} | — | — | — | — | — |"
    fin = m["final"]
    return (f"| {label} | {fin['rmse']:.3f} | {fin['mae']:.3f} | "
            f"{fin['bias']:+.3f} | {fin['r2']:.3f} | {m['best_rmse']:.3f} |")


def bins_row(label, m):
    if m is None: return None
    pb = m["final"].get("per_bin", {})
    cells = []
    for b in ["0-2m", "2-5m", "5-10m", "10-20m", "20-35m"]:
        if b in pb:
            cells.append(f"{pb[b]['rmse']:.2f}")
        else:
            cells.append("—")
    return f"| {label} | " + " | ".join(cells) + " |"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    runs_root = Path(args.runs)
    cases = [
        ("Agia Napa — baseline", runs_root / "baseline_an"),
        ("Agia Napa — Stumpf",   runs_root / "stumpf_an"),
        ("Puck Lagoon — baseline", runs_root / "baseline_pl"),
        ("Puck Lagoon — Stumpf",   runs_root / "stumpf_pl"),
    ]
    loaded = [(lab, load(d)) for lab, d in cases]

    md = []
    md.append("# Sentinel-2 SDB — Baseline vs Stumpf-augmented (MagicBathyNet)\n")
    md.append("## Test metrics (depths in meters)\n")
    md.append("| Run | RMSE | MAE | Bias | R² | Best RMSE |")
    md.append("|---|---:|---:|---:|---:|---:|")
    for lab, m in loaded:
        md.append(fmt_row(lab, m))
    md.append("")
    md.append("## Per-depth-bin RMSE (meters)\n")
    md.append("| Run | 0–2 m | 2–5 m | 5–10 m | 10–20 m | 20–35 m |")
    md.append("|---|---:|---:|---:|---:|---:|")
    for lab, m in loaded:
        row = bins_row(lab, m)
        if row: md.append(row)
    md.append("")

    # Compute improvement vs baseline per site
    def get_rmse(label_substr, stumpf):
        for lab, m in loaded:
            if m is None: continue
            if label_substr in lab and ("Stumpf" if stumpf else "baseline") in lab:
                return m["final"]["rmse"]
        return None

    md.append("## Stumpf vs baseline improvement\n")
    for site in ["Agia Napa", "Puck Lagoon"]:
        b = get_rmse(site, False); s = get_rmse(site, True)
        if b is not None and s is not None and b > 0:
            delta = b - s; pct = 100 * delta / b
            md.append(f"- **{site}**: baseline RMSE {b:.3f} m → Stumpf RMSE {s:.3f} m "
                      f"(Δ {delta:+.3f} m, {pct:+.1f}%)")
        else:
            md.append(f"- **{site}**: incomplete")
    md.append("")

    Path(args.out).write_text("\n".join(md))
    print("[wrote]", args.out)

    # combined scatter (2x2)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 2, figsize=(9, 9))
        for ax, (lab, m) in zip(axes.ravel(), loaded):
            if m is None:
                ax.set_title(f"{lab}\n(no data)")
                ax.axis("off"); continue
            run_dir = runs_root / (
                "baseline_an" if "Agia Napa — baseline" in lab else
                "stumpf_an"   if "Agia Napa — Stumpf"   in lab else
                "baseline_pl" if "Puck Lagoon — baseline" in lab else
                "stumpf_pl"
            )
            npz = np.load(run_dir / "test_predictions.npz")
            pred = npz["pred"]; gt = npz["gt"]
            ax.scatter(gt, pred, s=2, alpha=0.3)
            lo = float(min(gt.min(), pred.min())); hi = float(max(gt.max(), pred.max()))
            ax.plot([lo, hi], [lo, hi], 'r--', lw=1)
            ax.set_xlabel("Reference depth (m)"); ax.set_ylabel("Predicted (m)")
            ax.set_title(f"{lab}\nRMSE={m['final']['rmse']:.2f}  R²={m['final']['r2']:.2f}")
        fig.tight_layout(); fig.savefig(runs_root / "scatter_all.png", dpi=130)
        plt.close(fig)
        print("[wrote]", runs_root / "scatter_all.png")
    except Exception as e:
        print("[warn] combined scatter failed:", e)

if __name__ == "__main__":
    main()
