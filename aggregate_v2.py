"""
Aggregate v2: pulls in PhysSDB results AND cross-site evals; writes a unified
markdown summary including uncertainty metrics where available.
"""
import argparse, json
from pathlib import Path
import numpy as np

ORDER = [
    ("Agia Napa — Baseline (RGB)",       "baseline_an"),
    ("Agia Napa — Stumpf",               "stumpf_an"),
    ("Agia Napa — PhysSDB",              "physsdb_an"),
    ("Puck Lagoon — Baseline (RGB)",     "baseline_pl"),
    ("Puck Lagoon — Stumpf",             "stumpf_pl"),
    ("Puck Lagoon — PhysSDB",            "physsdb_pl"),
]

CROSS_ORDER = [
    ("AN→PL  Baseline",   "cross_baseline_an2pl"),
    ("AN→PL  Stumpf",     "cross_stumpf_an2pl"),
    ("AN→PL  PhysSDB",    "cross_physsdb_an2pl"),
]


def load(p: Path):
    f = p / "metrics.json"
    if not f.exists(): return None
    return json.loads(f.read_text())


def f(x, n=3):
    try: return f"{float(x):.{n}f}"
    except Exception: return "—"


def row(label, m, with_unc=False):
    if m is None:
        return f"| {label} | — | — | — | — |" + (" — | — |" if with_unc else "")
    fin = m["final"]
    base = (f"| {label} | {f(fin['rmse'])} | {f(fin['mae'])} | "
            f"{('+' if fin['bias']>=0 else '')}{f(fin['bias'])} | {f(fin['r2'])} |")
    if with_unc:
        ece = fin.get("ece", None)
        cov = fin.get("pi_coverage_95", None)
        base += f" {f(ece)} | {f(cov)} |"
    return base


def bin_row(label, m):
    if m is None: return None
    pb = m["final"].get("per_bin", {})
    cells = [pb.get(b, {}).get("rmse", None) for b in ["0-2m","2-5m","5-10m","10-20m","20-35m"]]
    return f"| {label} | " + " | ".join(f(c, 2) for c in cells) + " |"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    R = Path(args.runs)

    md = []
    md.append("# Sentinel-2 SDB — Full comparison (baseline / Stumpf / PhysSDB)\n")

    # ---- Main in-site table with uncertainty columns ----
    md.append("## In-site test metrics (depths in meters)\n")
    md.append("| Run | RMSE | MAE | Bias | R² | ECE (m) | 95% PI cov |")
    md.append("|---|---:|---:|---:|---:|---:|---:|")
    runs = [(lab, load(R / d)) for lab, d in ORDER]
    for lab, m in runs:
        md.append(row(lab, m, with_unc=True))
    md.append("")

    # ---- Per-depth-bin RMSE ----
    md.append("## Per-depth-bin RMSE (meters)\n")
    md.append("| Run | 0–2 m | 2–5 m | 5–10 m | 10–20 m | 20–35 m |")
    md.append("|---|---:|---:|---:|---:|---:|")
    for lab, m in runs:
        r = bin_row(lab, m)
        if r: md.append(r)
    md.append("")

    # ---- Improvement summary per site ----
    def get(label_part, m_or_none):
        return m_or_none["final"]["rmse"] if m_or_none else None

    rmse_an_base = next((m["final"]["rmse"] for lab, m in runs if m and "Agia Napa" in lab and "Baseline" in lab), None)
    rmse_an_phys = next((m["final"]["rmse"] for lab, m in runs if m and "Agia Napa" in lab and "PhysSDB" in lab), None)
    rmse_pl_base = next((m["final"]["rmse"] for lab, m in runs if m and "Puck Lagoon" in lab and "Baseline" in lab), None)
    rmse_pl_phys = next((m["final"]["rmse"] for lab, m in runs if m and "Puck Lagoon" in lab and "PhysSDB" in lab), None)

    md.append("## PhysSDB vs Baseline (RGB U-Net) — headline improvement\n")
    for site, b, p in [("Agia Napa", rmse_an_base, rmse_an_phys),
                       ("Puck Lagoon", rmse_pl_base, rmse_pl_phys)]:
        if b is not None and p is not None and b > 0:
            delta = b - p; pct = 100 * delta / b
            md.append(f"- **{site}**: baseline RMSE **{b:.3f} m** → PhysSDB RMSE **{p:.3f} m** "
                      f"(Δ {('+' if delta>0 else '')}{delta:.3f} m, **{'+' if pct>0 else ''}{pct:.1f}%**)")
        else:
            md.append(f"- **{site}**: incomplete")
    md.append("")

    # ---- Cross-site table ----
    cross = [(lab, load(R / d)) for lab, d in CROSS_ORDER]
    if any(m is not None for _, m in cross):
        md.append("## Cross-site generalization (train Agia Napa → test Puck Lagoon)\n")
        md.append("| Method | RMSE | MAE | Bias | R² |")
        md.append("|---|---:|---:|---:|---:|")
        for lab, m in cross:
            md.append(row(lab, m, with_unc=False))
        md.append("")

        # gap analysis vs in-site Puck Lagoon
        md.append("### Cross-site generalization gap = cross_RMSE − in-site Puck Lagoon RMSE")
        in_site = {lab.split(" — ")[-1].split(" ")[0]: m["final"]["rmse"]
                   for lab, m in runs if m and "Puck Lagoon" in lab}
        for cross_lab, cm in cross:
            if cm is None: continue
            method = cross_lab.split()[-1]
            if method in in_site:
                gap = cm["final"]["rmse"] - in_site[method]
                md.append(f"- **{method}**: cross RMSE {cm['final']['rmse']:.3f} m, "
                          f"in-site PL RMSE {in_site[method]:.3f} m, **gap {gap:+.3f} m**")
        md.append("")

    md.append("## Method summary\n")
    md.append("- **Baseline (RGB)**: MagicBathyNet U-Net on 3-channel RGB, masked RMSE loss.")
    md.append("- **Stumpf**: same U-Net + 2 extra input channels (log-ratios of B/G and B/R).")
    md.append("- **PhysSDB (ours)**: same trunk → 7-channel output (depth, a, b_b, ρR/G/B, log σ); "
              "trained with heteroscedastic NLL + differentiable Lee-1998-style reflectance "
              "reconstruction loss; emits per-pixel uncertainty σ.")
    md.append("")

    Path(args.out).write_text("\n".join(md))
    print("[wrote]", args.out)

    # ---- Combined figure (3 columns: Baseline / Stumpf / PhysSDB, 2 rows: AN / PL) ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 3, figsize=(13, 9))
        ds = [
            ("AN baseline",  R / "baseline_an"),
            ("AN Stumpf",    R / "stumpf_an"),
            ("AN PhysSDB",   R / "physsdb_an"),
            ("PL baseline",  R / "baseline_pl"),
            ("PL Stumpf",    R / "stumpf_pl"),
            ("PL PhysSDB",   R / "physsdb_pl"),
        ]
        for ax, (lab, p) in zip(axes.ravel(), ds):
            npz = p / "test_predictions.npz"
            jf  = p / "metrics.json"
            if not (npz.exists() and jf.exists()):
                ax.set_title(f"{lab}\n(no data)"); ax.axis("off"); continue
            d = np.load(npz); m = json.loads(jf.read_text())["final"]
            ax.scatter(d["gt"], d["pred"], s=2, alpha=0.3)
            lo, hi = float(min(d["gt"].min(), d["pred"].min())), \
                     float(max(d["gt"].max(), d["pred"].max()))
            ax.plot([lo, hi], [lo, hi], "r--", lw=1)
            t = f"{lab}\nRMSE={m['rmse']:.2f}m  R²={m['r2']:.2f}"
            if "ece" in m: t += f"  ECE={m['ece']:.2f}  PI95={m['pi_coverage_95']:.2f}"
            ax.set_title(t); ax.set_xlabel("ref (m)"); ax.set_ylabel("pred (m)")
        fig.tight_layout(); fig.savefig(R / "scatter_all_v2.png", dpi=130); plt.close(fig)
        print("[wrote]", R / "scatter_all_v2.png")
    except Exception as e:
        print("[warn] combined figure:", e)


if __name__ == "__main__":
    main()
