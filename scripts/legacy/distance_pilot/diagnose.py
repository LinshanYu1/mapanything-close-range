"""Inspect RGB-only predictions; sensor depth is used after inference only."""

from pathlib import Path
import json
import numpy as np
import torch
from PIL import Image
from scipy.io import loadmat
from mapanything.models import MapAnything
from mapanything.utils.image import load_images

base = Path(__file__).resolve().parents[2]
out = base / "output/distance_pilot_20"
m = json.loads((out / "manifest.json").read_text())
views = load_images(
    [str(base / r["input_image"]) for r in m["frames"]],
    resize_mode="fixed_size",
    size=(518, 294),
)
model = (
    MapAnything.from_pretrained(
        "facebook/map-anything", revision="a1d87e9086706fb9974f3be5a3e3a0ca5401c5aa"
    )
    .to("mps")
    .eval()
)
with torch.inference_mode():
    preds = model.infer(
        views,
        memory_efficient_inference=True,
        minibatch_size=1,
        use_amp=False,
        apply_mask=True,
        mask_edges=True,
        apply_confidence_mask=False,
    )
rows = []
for r, p in zip(m["frames"], preds):

    def arr(k):
        return p[k].detach().cpu().numpy()[0]

    xyz = arr("pts3d")
    cam = arr("pts3d_cam")
    z = arr("depth_z").squeeze()
    mask = arr("mask").squeeze().astype(bool)
    q = np.rint(r["points_model"]).astype(int)
    u, v = q.T
    gt = (
        np.array(
            Image.open(base / r["depth_file"]).resize(
                (518, 294), Image.Resampling.NEAREST
            )
        )
        / 1000
    )
    good = mask & (gt > 0) & np.isfinite(z) & (z > 0)
    row = {
        "frame": r["frame"],
        "qc": r["qc_pass"],
        "point_valid": mask[v, u].tolist(),
        "predicted_point_z_m": z[v, u].tolist(),
        "sensor_point_z_m": r["depth_at_points_m"],
        "scale": arr("metric_scaling_factor").tolist(),
        "median_dense_depth_ratio": float(np.median(z[good] / gt[good])),
        "valid_dense_pixels": int(good.sum()),
        "camera_distance_m": float(np.linalg.norm(cam[v, u][0] - cam[v, u][1])),
        "world_distance_m": float(np.linalg.norm(xyz[v, u][0] - xyz[v, u][1])),
        "predicted_intrinsics": arr("intrinsics").tolist(),
    }
    rows.append(row)
    np.savez_compressed(
        out / f'diagnostic_{r["frame"]:04d}.npz',
        depth_z=z,
        mask=mask,
        points_camera=cam[v, u],
        points_world=xyz[v, u],
    )
(out / "diagnostics.json").write_text(json.dumps(rows, indent=2))
print(json.dumps(rows, indent=2))
