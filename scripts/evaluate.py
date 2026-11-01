"""Score a saved reference point map using phone or sensor-derived annotations."""

import argparse, csv
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from close_range.common import read, save, resolve, phone_distance


def score(xyz, mask, points, target):
    q = np.rint(points).astype(int)
    h, w = mask.shape
    inside = bool((q >= 0).all() and (q[:, 0] < w).all() and (q[:, 1] < h).all())
    valid = (
        inside
        and bool(mask[q[:, 1], q[:, 0]].all())
        and bool(np.isfinite(xyz[q[:, 1], q[:, 0]]).all())
    )
    pred = (
        float(np.linalg.norm(xyz[q[0, 1], q[0, 0]] - xyz[q[1, 1], q[1, 0]]))
        if valid
        else None
    )
    err = abs(pred - target) if valid else None
    return dict(
        prediction_valid=valid,
        reference_inch=target / 0.0254,
        predicted_inch=pred / 0.0254 if valid else None,
        reference_cm=target * 100,
        predicted_cm=pred * 100 if valid else None,
        absolute_error_cm=err * 100 if valid else None,
        relative_error_pct=err / target * 100 if valid else None,
        success_10pct=bool(valid and err / target <= 0.1),
        reason=(
            "" if valid else ("outside_image" if not inside else "masked_or_nonfinite")
        ),
    )


def evaluate_run(m, out):
    ann = read(resolve(m["annotation"]))
    run = read(out / "run.json")
    if (
        run["scene_id"] != ann["scene_id"]
        or run["reference_view_id"] != ann["reference_view_id"]
        or run["reference_sha256"] != ann["image_sha256"]
    ):
        raise ValueError("Prediction/annotation mismatch")
    if m["source"] == "phone":
        distance, _ = phone_distance(m["scene_id"], m.get("distance_config"))
        if any(
            not np.isclose(pair["distance_m"], distance * 0.0254, rtol=0, atol=1e-10)
            for pair in ann["pairs"]
        ):
            raise ValueError(
                "distances.conf changed after annotation. Re-run scripts/phone/annotate.py --overwrite to confirm endpoints and measurement."
            )
    pred = np.load(out / "reference_prediction.npz")
    rows = []
    for pair in ann["pairs"]:
        if not pair["label_valid"]:
            continue
        target = pair["distance_m"]
        if not np.isfinite(target) or target <= 0:
            raise ValueError("Invalid target distance")
        rows.append(
            dict(
                pair_id=pair["id"],
                label_source=pair["label_source"],
                **score(pred["pts3d"], pred["mask"], pair["points_xy"], target)
            )
        )
    if not rows:
        raise ValueError("No eligible labels")
    with (out / "metrics.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    summary = dict(
        eligible_pairs=len(rows),
        valid_predictions=sum(r["prediction_valid"] for r in rows),
        success_rate=sum(r["success_10pct"] for r in rows) / len(rows),
        results=rows,
        note="Pilot only; no ground-truth scale alignment. Invalid predictions count as failures.",
    )
    save(out / "metrics.json", summary)
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", default="data/manifests/phone/scene-001.json")
    p.add_argument("--run", required=True)
    a = p.parse_args()
    summary = evaluate_run(read(resolve(a.manifest)), resolve(a.run))
    for row in summary["results"]:
        print(row)


if __name__ == "__main__":
    main()
