"""Run MapAnything with RGB or RGB + known intrinsics and score a multi-view pilot. Needs official package."""

from pathlib import Path
import argparse, json, csv
import numpy as np

BASE = Path(__file__).resolve().parents[2]
p = argparse.ArgumentParser()
p.add_argument("--model", default="facebook/map-anything")
p.add_argument("--device", default="cuda")
p.add_argument(
    "--manifest", type=Path, default=BASE / "output/distance_pilot_20/manifest.json"
)
p.add_argument("--known-intrinsics", action="store_true")
p.add_argument("--output", type=Path)
p.add_argument("--revision", default="a1d87e9086706fb9974f3be5a3e3a0ca5401c5aa")
a = p.parse_args()
import torch
from mapanything.models import MapAnything
from mapanything.utils.image import load_images

if a.device == "cuda" and not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA unavailable. Run on a GPU machine, or explicitly request --device cpu (slow)."
    )
out = a.output or a.manifest.resolve().parent
out.mkdir(parents=True, exist_ok=True)
m = json.loads(a.manifest.read_text())
files = [str(BASE / r["input_image"]) for r in m["frames"]]
views = load_images(files, resize_mode="fixed_size", size=(518, 294))
assert len(views) == len(m["frames"])
for view in views:
    assert tuple(view["img"].shape[-2:]) == (294, 518), "Unexpected image transform"
if a.known_intrinsics:
    from scipy.io import loadmat
    from PIL import Image

    calibrated = []
    for r, view in zip(m["frames"], views):
        depth_path = BASE / r["depth_file"]
        K = loadmat(depth_path.parent.parent / "meta" / f"{r['frame']:04d}.mat")[
            "intrinsic_matrix"
        ].astype(np.float64)
        with Image.open(
            depth_path.parent.parent / "rgb" / f"{r['frame']:04d}.png"
        ) as image:
            w, h = image.size
        # Same half-pixel resize transform as the point annotations.
        A = np.array(
            [
                [518 / w, 0, (518 / w - 1) / 2],
                [0, 294 / h, (294 / h - 1) / 2],
                [0, 0, 1],
            ]
        )
        K = A @ K
        view["intrinsics"] = torch.tensor(K, dtype=torch.float32)[None]
        calibrated.append(dict(frame=r["frame"], intrinsics=K.tolist()))
    (out / "input_intrinsics.json").write_text(json.dumps(calibrated, indent=2))
print(f"Loading {a.model} on {a.device}; {len(views)} RGB views", flush=True)
model = MapAnything.from_pretrained(a.model, revision=a.revision).to(a.device).eval()
print("Running inference...", flush=True)
with torch.inference_mode():
    preds = model.infer(
        views,
        memory_efficient_inference=True,
        minibatch_size=1,
        use_amp=a.device.startswith("cuda"),
        apply_mask=True,
        mask_edges=True,
        apply_confidence_mask=False,
    )
assert len(preds) == len(views)
rows = []
for r, pred in zip(m["frames"], preds):
    xyz = pred["pts3d"][0].detach().cpu().numpy()
    mask = pred["mask"][0].detach().cpu().numpy().squeeze()
    assert xyz.shape == (294, 518, 3)
    q = np.rint(r["points_model"]).astype(int)
    valid = bool(np.all(q >= 0) and np.all(q[:, 0] < 518) and np.all(q[:, 1] < 294))
    distance = None
    if valid:
        points = xyz[q[:, 1], q[:, 0]]
        valid = bool(np.all(mask[q[:, 1], q[:, 0]]) and np.isfinite(points).all())
        if valid:
            distance = float(np.linalg.norm(points[0] - points[1]))
    ref = m["reference_distance_m"]
    err = abs(distance - ref) if distance is not None else None
    eligible = r["qc_pass"]
    passed = bool(
        eligible and err is not None and err / ref <= m["success_relative_error"]
    )
    rows.append(
        dict(
            frame=r["frame"],
            reference_cm=ref * 100,
            label_qc_pass=eligible,
            qc_reasons=";".join(r["qc_reasons"]),
            prediction_valid=valid,
            predicted_cm=distance * 100 if distance is not None else None,
            absolute_error_cm=err * 100 if err is not None else None,
            relative_error=err / ref if err is not None else None,
            success=passed,
        )
    )
with (out / "baseline.csv").open("w") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys())
    w.writeheader()
    w.writerows(rows)
eligible = [r for r in rows if r["label_qc_pass"]]
valid = [r for r in eligible if r["prediction_valid"]]
summary = dict(
    model=a.model,
    revision=a.revision,
    known_intrinsics=a.known_intrinsics,
    device=a.device,
    torch_version=torch.__version__,
    joint_input_frames=len(views),
    independent_scenes=1,
    qc_eligible=len(eligible),
    valid_predictions=len(valid),
    successful=sum(r["success"] for r in eligible),
    success_rate_including_prediction_failures=(
        sum(r["success"] for r in eligible) / len(eligible) if eligible else None
    ),
    mean_absolute_error_cm=(
        float(np.mean([r["absolute_error_cm"] for r in valid])) if valid else None
    ),
    note="Sensor reference; GT-assisted point correspondences; No GT depth, poses or distance passed to model; known_intrinsics indicates whether calibration was supplied. No metric scale fitting. Completed inference; first-frame sensor reference is used for scoring.",
)
(out / "baseline_summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
