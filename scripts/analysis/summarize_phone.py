
import csv
import math
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUTPUTS = ROOT / "outputs"
DEST = OUTPUTS / "phone_baseline_analysis"
DEST.mkdir(parents=True, exist_ok=True)

rows = []
missing = []

for n in range(21, 41):
    scene = f"scene-{n:03d}"

    for mode in ("single", "multi"):
        path = OUTPUTS / f"phone_{n:03d}_m0_{mode}" / "metrics.csv"

        if not path.exists():
            missing.append(f"{scene}/{mode}")
            continue

        with path.open(newline="") as f:
            results = list(csv.DictReader(f))

        if len(results) != 1:
            raise ValueError(
                f"{path}: expected one pair, found {len(results)}"
            )

        r = results[0]
        valid = r["prediction_valid"].lower() == "true"
        gt = float(r["reference_cm"])

        pred = float(r["predicted_cm"]) if valid else math.nan
        abs_err = abs(pred - gt) if valid else math.nan
        rel_err = abs_err / gt * 100 if valid else math.nan
        ratio = pred / gt if valid else math.nan

        rows.append({
            "scene": scene,
            "mode": mode,
            "valid": valid,
            "gt_cm": gt,
            "pred_cm": pred,
            "abs_error_cm": abs_err,
            "relative_error_pct": rel_err,
            "scale_ratio": ratio,
            "success_10pct": valid and rel_err <= 10,
            "reason": r.get("reason", ""),
        })

if missing:
    raise RuntimeError(
        "Missing results:\n" + "\n".join(missing)
    )

detail_path = DEST / "phone_baseline_summary.csv"

with detail_path.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)

stats = []

for mode in ("single", "multi"):
    subset = [r for r in rows if r["mode"] == mode]
    valid = [r for r in subset if r["valid"]]

    stats.append({
        "mode": mode,
        "total": len(subset),
        "valid": len(valid),
        "valid_rate_pct": 100 * len(valid) / len(subset),
        "mae_cm_valid": (
            statistics.mean(r["abs_error_cm"] for r in valid)
            if valid else math.nan
        ),
        "mean_relative_error_pct_valid": (
            statistics.mean(r["relative_error_pct"] for r in valid)
            if valid else math.nan
        ),
        "median_scale_ratio_valid": (
            statistics.median(r["scale_ratio"] for r in valid)
            if valid else math.nan
        ),
        "success_10pct_all": (
            100 * sum(r["success_10pct"] for r in subset) / len(subset)
        ),
    })

stats_path = DEST / "phone_baseline_statistics.csv"

with stats_path.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(stats[0]))
    writer.writeheader()
    writer.writerows(stats)

print(f"\nSaved: {detail_path}")
print(f"Saved: {stats_path}")

print("\n=== BASELINE STATISTICS ===")
for s in stats:
    print(f"\nMode: {s['mode']}")
    print(f"  Valid: {s['valid']}/{s['total']}")
    print(f"  Valid rate: {s['valid_rate_pct']:.1f}%")
    print(f"  MAE (valid): {s['mae_cm_valid']:.2f} cm")
    print(
        "  Mean relative error (valid): "
        f"{s['mean_relative_error_pct_valid']:.2f}%"
    )
    print(
        "  Median scale ratio (valid): "
        f"{s['median_scale_ratio_valid']:.3f}"
    )
    print(
        "  Success@10% (all): "
        f"{s['success_10pct_all']:.1f}%"
    )
