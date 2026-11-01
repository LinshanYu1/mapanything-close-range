"""Build a ten-view RGB-only pilot; depth is used ONLY for reference/QC."""

from pathlib import Path
import argparse, json, tarfile, csv
import numpy as np
from scipy.io import loadmat
from PIL import Image, ImageDraw

BASE = Path(__file__).resolve().parents[2]
p = argparse.ArgumentParser()
p.add_argument("--points", type=int, nargs=4, default=[610, 198, 800, 330])
p.add_argument("--archive", default="/private/tmp/graspnet-scene-prefix.tar")
p.add_argument("--count", type=int, choices=[10, 20], default=10)
p.add_argument("--tolerance-mm", type=float, default=10)
a = p.parse_args()
root = BASE / "data/graspnet_audit/scene_0017/realsense"
out = BASE / ("output/distance_pilot" if a.count == 10 else "output/distance_pilot_20")
out.mkdir(parents=True, exist_ok=True)
frames = list(range(0, 240, 240 // a.count))
# Recover only missing sampled files from the already downloaded archive.
wanted = {
    f"scenes/scene_0017/realsense/{kind}/{i:04d}.{ext}"
    for i in frames
    for kind, ext in [
        ("rgb", "png"),
        ("depth", "png"),
        ("label", "png"),
        ("meta", "mat"),
    ]
    if not (root / kind / f"{i:04d}.{ext}").exists()
}
if wanted:
    with tarfile.open(a.archive) as t:
        try:
            for m in t:
                if m.name in wanted:
                    dst = root / Path(*Path(m.name).parts[3:])
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    dst.write_bytes(t.extractfile(m).read())
                    wanted.remove(m.name)
                    if not wanted:
                        break
        except tarfile.ReadError:
            pass
    if wanted:
        raise RuntimeError(f"Missing files: {sorted(wanted)}")


def read(i):
    m = loadmat(root / f"meta/{i:04d}.mat")
    return (
        m["intrinsic_matrix"],
        np.asarray(Image.open(root / f"depth/{i:04d}.png"))
        / float(m["factor_depth"].item()),
        np.asarray(Image.open(root / f"label/{i:04d}.png")),
    )


def backproject(uv, z, K):
    return (np.linalg.inv(K) @ np.c_[uv, np.ones(len(uv))].T).T * z[:, None]


K, d, seg = read(0)
uv = np.array(a.points).reshape(2, 2)
if np.any(uv < 0) or np.any(uv[:, 0] >= d.shape[1]) or np.any(uv[:, 1] >= d.shape[0]):
    raise ValueError("Points outside image")
z = d[uv[:, 1], uv[:, 0]]
ids = seg[uv[:, 1], uv[:, 0]]
if np.any(z <= 0) or np.any(ids == 0) or ids[0] == ids[1]:
    raise ValueError("Choose valid depth points on two different objects")
P = backproject(uv, z, K)
distance = float(np.linalg.norm(P[0] - P[1]))
poses = np.load(root / "camera_poses.npy")
rows = []
canvas = Image.new("RGB", (1280, ((a.count + 1) // 2) * 385), "white")
cd = ImageDraw.Draw(canvas)
for j, i in enumerate(frames):
    K, d, seg = read(i)
    pc = (np.linalg.inv(poses[i]) @ np.c_[P, np.ones(2)].T).T[:, :3]
    projected = (K @ pc.T).T
    points = projected[:, :2] / projected[:, 2:]
    q = np.rint(points).astype(int)
    rgb = Image.open(root / f"rgb/{i:04d}.png").convert("RGB")
    w, h = rgb.size
    reason = []
    raw = None
    residual = None
    zz = None
    inside = bool(
        np.all(pc[:, 2] > 0)
        and np.all(q >= 0)
        and np.all(q[:, 0] < w)
        and np.all(q[:, 1] < h)
    )
    if not inside:
        reason.append("outside_image_or_behind_camera")
    else:
        zz = d[q[:, 1], q[:, 0]]
        if np.any(zz <= 0):
            reason.append("missing_depth")
        if np.any(seg[q[:, 1], q[:, 0]] != ids):
            reason.append("object_label_mismatch")
        if np.all(zz > 0):
            residual = ((zz - pc[:, 2]) * 1000).tolist()
            raw = float(np.linalg.norm(np.diff(backproject(q, zz, K), axis=0)))
            if max(abs(x) for x in residual) > a.tolerance_mm:
                reason.append("depth_inconsistent_possible_occlusion")
    # Explicit resize fixes coordinate mapping; no hidden cropping permitted downstream.
    resized = rgb.resize((518, 294), Image.Resampling.LANCZOS)
    inp = out / "rgb" / f"{i:04d}.png"
    inp.parent.mkdir(exist_ok=True)
    resized.save(inp)
    mapped = (points + 0.5) * np.array([518 / w, 294 / h]) - 0.5
    row = dict(
        frame=i,
        input_image=str(inp.relative_to(BASE)),
        points_original=points.tolist(),
        points_model=mapped.tolist(),
        qc_pass=not reason,
        qc_reasons=reason,
        raw_depth_distance_m=raw,
        depth_residual_mm=residual,
        depth_file=str((root / f"depth/{i:04d}.png").relative_to(BASE)),
        depth_at_points_m=zz.tolist() if zz is not None else None,
        pixel_indices=q.tolist(),
    )
    rows.append(row)
    dr = ImageDraw.Draw(rgb)
    for n, (u, v) in enumerate(q):
        if 0 <= u < w and 0 <= v < h:
            dr.ellipse((u - 10, v - 10, u + 10, v + 10), outline="red", width=4)
            dr.text((u + 14, v), str(n + 1), fill="red")
    ann = out / "annotated"
    ann.mkdir(exist_ok=True)
    rgb.save(ann / f"{i:04d}.jpg")
    rgb.thumbnail((640, 360))
    x = j % 2 * 640
    y = j // 2 * 385
    canvas.paste(rgb, (x, y))
    cd.text(
        (x + 5, y + 362),
        f"{i:04d}: " + ("QC PASS" if not reason else ", ".join(reason)),
        fill="black",
    )
manifest = dict(
    scene="scene_0017",
    camera="realsense",
    reference_distance_m=distance,
    reference="Frame 0000 sensor depth, NOT independent metrology",
    qc_tolerance_mm=a.tolerance_mm,
    success_relative_error=0.10,
    preprocessing="Explicit RGB resize 1280x720 to 518x294; half-pixel coordinate mapping; slight aspect change",
    frames=rows,
)
(out / "manifest.json").write_text(json.dumps(manifest, indent=2))
canvas.save(out / "annotated.jpg")

for start in range(0, a.count, 10):
    canvas.crop((0, start // 2 * 385, 1280, min(start + 10, a.count) // 2 * 385)).save(
        out / f"contact_sheet_{start//10+1}.jpg"
    )
with (out / "measurements.csv").open("w") as f:
    writer = csv.writer(f)
    writer.writerow(
        ["frame", "A_depth_mm", "B_depth_mm", "distance_cm", "QC", "reason"]
    )
    for r in rows:
        depths = r["depth_at_points_m"]
        writer.writerow(
            [
                r["frame"],
                depths[0] * 1000 if depths else "",
                depths[1] * 1000 if depths else "",
                (
                    r["raw_depth_distance_m"] * 100
                    if r["raw_depth_distance_m"] is not None
                    else ""
                ),
                r["qc_pass"],
                ";".join(r["qc_reasons"]),
            ]
        )
print(
    f'Reference: {distance*100:.3f} cm; QC passed {sum(r["qc_pass"] for r in rows)}/{a.count}; output: {out}'
)
