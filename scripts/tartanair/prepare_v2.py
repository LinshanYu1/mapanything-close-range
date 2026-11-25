"""Prepare TartanAir V1 close-range multi-view training groups."""

import argparse
import csv
import io
import sys
import zipfile
from collections import Counter
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import binary_erosion, uniform_filter
from scipy.spatial.transform import Rotation

sys.path.insert(
    0, str(Path(__file__).resolve().parents[2] / "src")
)

from close_range.common import ROOT, save, sha, portable_path
from mapanything.utils.cropping import crop_resize_if_necessary
from mapanything.utils.image import find_closest_aspect_ratio


RAW_ROOT = (
    Path.home()
    / "CS541/mapanything-original-data/tartanair-v1"
)

ENV = "japanesealley"
DIFFICULTY = "Easy"

K = np.array([
    [320., 0., 320.],
    [0., 320., 240.],
    [0., 0., 1.]
])

K_INV = np.linalg.inv(K)

# OpenCV camera coordinates (right, down, forward)
# to NED camera coordinates (forward, right, down).
CV_TO_NED = np.array([
    [0., 0., 1.],
    [1., 0., 0.],
    [0., 1., 0.]
])

BINS = [
    (0.05, 0.10),
    (0.10, 0.20),
    (0.20, 0.40),
    (0.40, 0.60),
    (0.60, 1.00),
]


class TartanAir:

    def __init__(self, root):
        self.rgb_zip = zipfile.ZipFile(
            root / f"{ENV}_{DIFFICULTY}_image_left.zip"
        )
        self.depth_zip = zipfile.ZipFile(
            root / f"{ENV}_{DIFFICULTY}_depth_left.zip"
        )
        self.seg_zip = zipfile.ZipFile(
            root / f"{ENV}_{DIFFICULTY}_seg_left.zip"
        )

    def prefix(self, trajectory):
        return f"{ENV}/{DIFFICULTY}/{trajectory}"

    def rgb(self, trajectory, frame):
        name = (
            f"{self.prefix(trajectory)}/image_left/"
            f"{frame:06d}_left.png"
        )
        return Image.open(
            io.BytesIO(self.rgb_zip.read(name))
        ).convert("RGB")

    @lru_cache(maxsize=16)
    def depth(self, trajectory, frame):
        name = (
            f"{self.prefix(trajectory)}/depth_left/"
            f"{frame:06d}_left_depth.npy"
        )
        return np.load(
            io.BytesIO(self.depth_zip.read(name)),
            allow_pickle=False
        ).astype(np.float32)

    def seg(self, trajectory, frame):
        name = (
            f"{self.prefix(trajectory)}/seg_left/"
            f"{frame:06d}_left_seg.npy"
        )
        return np.load(
            io.BytesIO(self.seg_zip.read(name)),
            allow_pickle=False
        )

    @lru_cache(maxsize=6)
    def poses(self, trajectory):
        name = f"{self.prefix(trajectory)}/pose_left.txt"
        return np.loadtxt(
            io.BytesIO(self.rgb_zip.read(name))
        )


def backproject(points, depth):
    points = np.asarray(points, dtype=int)
    x, y = points.T

    rays = np.c_[
        points, np.ones(len(points))
    ] @ K_INV.T

    return rays * depth[y, x, None]


def camera_rotation(pose):
    # Assumes pose quaternion order qx,qy,qz,qw
    # and camera-to-world orientation in NED.
    return (
        Rotation.from_quat(pose[3:]).as_matrix()
        @ CV_TO_NED
    )


def project_to_support(xyz, ref_pose, support_pose):
    r_ref = camera_rotation(ref_pose)
    r_sup = camera_rotation(support_pose)

    world = xyz @ r_ref.T + ref_pose[:3]

    local = (world - support_pose[:3]) @ r_sup

    z = local[:, 2]
    projected = local @ K.T

    with np.errstate(divide="ignore", invalid="ignore"):
        uv = projected[:, :2] / z[:, None]

    return uv, z


def visible(xyz, ref_pose, support_pose, support_depth):
    uv, z = project_to_support(
        xyz, ref_pose, support_pose
    )

    h, w = support_depth.shape

    inside = (
        np.isfinite(uv).all(axis=1)
        & np.isfinite(z)
        & (z > 0)
        & (uv[:, 0] >= 5)
        & (uv[:, 0] <= w - 6)
        & (uv[:, 1] >= 5)
        & (uv[:, 1] <= h - 6)
    )

    result = np.zeros(len(xyz), dtype=bool)
    idx = np.flatnonzero(inside)

    if not len(idx):
        return result

    pixels = np.rint(uv[idx]).astype(int)
    x, y = pixels.T

    measured = support_depth[y, x]

    tolerance = np.maximum(0.03, 0.03 * z[idx])

    result[idx] = (
        np.isfinite(measured)
        & (measured > 0)
        & (np.abs(measured - z[idx]) <= tolerance)
    )

    return result


def select_supports(data, trajectory, frame, xyz):
    poses = data.poses(trajectory)
    ref_pose = poses[frame]

    candidates = []

    for offset in range(-5, 6):

        if offset == 0:
            continue

        f = frame + offset

        if not 0 <= f < len(poses):
            continue

        pose = poses[f]

        baseline = np.linalg.norm(
            pose[:3] - ref_pose[:3]
        )

        if not 0.05 <= baseline <= 0.30:
            continue

        relative = (
            Rotation.from_quat(ref_pose[3:]).inv()
            * Rotation.from_quat(pose[3:])
        )

        angle = np.degrees(relative.magnitude())

        if angle > 25:
            continue

        mask = visible(
            xyz,
            ref_pose,
            pose,
            data.depth(trajectory, f)
        )

        overlap = float(mask.mean())

        if overlap < 0.50:
            continue

        candidates.append({
            "frame": f,
            "baseline_m": float(baseline),
            "rotation_deg": float(angle),
            "overlap": overlap,
            "mask": mask,
        })

    # Select two views with maximum shared visibility.
    best = None
    best_score = -1

    for i, a in enumerate(candidates):
        for b in candidates[i+1:]:

            shared = a["mask"] & b["mask"]
            score = float(shared.mean())

            if score > best_score:
                best_score = score
                best = (a, b)

    if best is None or best_score < 0.40:
        return None

    return best


def eligible_pixels(image, depth, seg):

    valid = (
        np.isfinite(depth)
        & (depth >= 0.3)
        & (depth <= 2.0)
    )

    interior = np.zeros_like(valid)

    for label in np.unique(seg):
        interior |= binary_erosion(
            (seg == label) & valid,
            iterations=3
        )

    # Reject sharp depth discontinuities.
    smooth = np.zeros_like(valid)

    center = depth[1:-1, 1:-1]

    smooth[1:-1, 1:-1] = (
        (np.abs(center - depth[:-2, 1:-1]) < 0.03)
        & (np.abs(center - depth[2:, 1:-1]) < 0.03)
        & (np.abs(center - depth[1:-1, :-2]) < 0.03)
        & (np.abs(center - depth[1:-1, 2:]) < 0.03)
    )

    rgb = np.asarray(image, dtype=np.float32)

    gray = rgb.mean(axis=2)

    bright = uniform_filter(gray, size=7) >= 25

    mask = interior & smooth & bright

    mask[:20, :] = False
    mask[-20:, :] = False
    mask[:, :20] = False
    mask[:, -20:] = False

    return mask


def sample_pairs(pixels, original, xyz, labels, rng):

    pairs = []
    used = set()

    if len(pixels) < 2:
        return pairs

    for lo, hi in BINS:

        count = 0

        # Prefer different-label pairs, then allow
        # same-label pairs to fill missing distance bins.
        for cross_label in [True, False]:

            for _ in range(100):

                if count >= 2:
                    break

                i = int(rng.integers(len(pixels)))

                distances = np.linalg.norm(
                    xyz - xyz[i], axis=1
                )

                valid = (
                    (distances >= lo)
                    & (distances < hi)
                )

                if cross_label:
                    valid &= labels != labels[i]
                else:
                    valid &= labels == labels[i]

                possible = np.flatnonzero(valid)

                if not len(possible):
                    continue

                j = int(rng.choice(possible))

                key = tuple(sorted([
                    tuple(original[i]),
                    tuple(original[j])
                ]))

                if key in used:
                    continue

                used.add(key)

                pairs.append({
                    "id": f"pair_{len(pairs)+1:03d}",
                    "points_xy": pixels[[i, j]].tolist(),
                    "source_points_xy": original[[i, j]].tolist(),
                    "distance_m": float(distances[j]),
                    "segmentation_labels": labels[[i, j]].tolist(),
                    "pair_type": (
                        "cross_label"
                        if cross_label
                        else "same_label"
                    ),
                    "distance_bin_m": [lo, hi],
                    "label_source": "synthetic_depth",
                    "label_valid": True,
                })

                count += 1

            if count >= 2:
                break

    return pairs


def prepare_scene(data, trajectory, frame, args):

    name = f"tartanair_v2_{ENV}_{trajectory}_{frame:06d}"

    out = ROOT / "data/processed/tartanair_v2" / name

    ann = (
        ROOT / "data/annotations/tartanair_v2"
        / f"{name}.json"
    )

    manifest = (
        ROOT / "data/manifests/tartanair_v2"
        / f"{name}.json"
    )

    if out.is_dir() and ann.is_file() and manifest.is_file():
        return "existing"

    if any(p.exists() for p in [out, ann, manifest]):
        raise RuntimeError(f"Partial output: {name}")

    rng = np.random.default_rng(args.seed + frame)

    image = data.rgb(trajectory, frame)
    depth = data.depth(trajectory, frame)
    seg = data.seg(trajectory, frame)

    if depth.shape != (480, 640):
        raise ValueError("Unexpected depth shape")

    if seg.shape != depth.shape or image.size != (640, 480):
        raise ValueError("RGB/depth/seg shape mismatch")

    mask = eligible_pixels(image, depth, seg)

    size = find_closest_aspect_ratio(
        image.width / image.height, 518
    )

    source_ids = np.arange(
        depth.size, dtype=np.int32
    ).reshape(depth.shape)

    processed, quantities = crop_resize_if_necessary(
        image,
        resolution=size,
        additional_quantities=[source_ids]
    )

    ids = np.asarray(quantities[0], dtype=np.int64)

    if ids.min() < 0 or ids.max() >= depth.size:
        raise ValueError("Invalid source-pixel mapping")

    sy, sx = np.divmod(ids, image.width)

    eligible = mask[sy, sx]

    pixels = np.argwhere(eligible)[:, ::-1]

    if len(pixels) < 100:
        return "insufficient_pixels"

    # Limit memory and computation on CPU.
    if len(pixels) > 8000:
        pixels = pixels[
            rng.choice(len(pixels), 8000, replace=False)
        ]

    x, y = pixels.T

    original = np.c_[sx[y, x], sy[y, x]]

    xyz = backproject(original, depth)

    supports = select_supports(
        data, trajectory, frame, xyz
    )

    if supports is None:
        return "no_supports"

    a, b = supports

    shared = a["mask"] & b["mask"]

    pixels = pixels[shared]
    original = original[shared]
    xyz = xyz[shared]

    ox, oy = original.T
    labels = seg[oy, ox]

    pairs = sample_pairs(
        pixels, original, xyz, labels, rng
    )

    if len(pairs) < args.min_pairs:
        return "insufficient_pairs"

    frames = [frame, a["frame"], b["frame"]]

    out.mkdir(parents=True)

    views = []

    for i, f in enumerate(frames):

        img = data.rgb(trajectory, f)

        dest = out / f"view_{i+1:03d}.png"

        crop_resize_if_necessary(
            img, resolution=size
        )[0].save(dest)

        views.append({
            "id": f"view_{i+1:03d}",
            "image": portable_path(dest),
            "sha256": sha(dest),
            "frame": f"{f:06d}",
            "trajectory": trajectory,
            "original": (
                f"{ENV}/{DIFFICULTY}/{trajectory}/"
                f"image_left/{f:06d}_left.png"
            ),
        })

    ann.parent.mkdir(parents=True, exist_ok=True)
    manifest.parent.mkdir(parents=True, exist_ok=True)

    save(ann, {
        "scene_id": name,
        "reference_view_id": views[0]["id"],
        "image_sha256": views[0]["sha256"],
        "coordinate_space": "model_input_pixels_xy",
        "pairs": pairs,
        "synthetic_reference": {
            "environment": ENV,
            "trajectory": trajectory,
            "frame": frame,
            "intrinsic": K.tolist(),
            "depth_unit": "meter",
        },
        "note": (
            "Synthetic depth-derived labels. "
            "Different segmentation IDs do not "
            "guarantee distinct physical instances."
        ),
    })

    save(manifest, {
        "schema_version": 1,
        "scene_id": name,
        "source": "tartanair",
        "original_scene": ENV,
        "trajectory": trajectory,
        "camera": "left",
        "views": views,
        "reference_index": 0,
        "model_size": list(size),
        "annotation": portable_path(ann),
        "preprocessing": (
            "Upstream crop/resize; "
            "nearest source-pixel mapping"
        ),
        "seed": args.seed,
        "support_geometry": [
            {
                k: v for k, v in s.items()
                if k != "mask"
            }
            for s in supports
        ],
    })

    previews = (
        ROOT / "data/previews/tartanair_v2" / name
    )

    previews.mkdir(parents=True, exist_ok=True)

    for pair in pairs:

        preview = processed.copy()
        draw = ImageDraw.Draw(preview)

        p1, p2 = map(tuple, pair["points_xy"])

        draw.line([p1, p2], fill="yellow", width=2)

        for letter, (x, y), color in [
            ("A", p1, "red"),
            ("B", p2, "cyan"),
        ]:
            draw.ellipse(
                (x-4, y-4, x+4, y+4),
                outline=color,
                width=2
            )
            draw.text((x+6, y), letter, fill=color)

        draw.text(
            (10, 10),
            f'{pair["distance_m"]*100:.2f} cm',
            fill="yellow"
        )

        preview.save(
            previews / f'{pair["id"]}.png'
        )

    print(
        f"Prepared {name}: "
        f"frames={frames}, pairs={len(pairs)}, "
        f"shared={shared.mean():.1%}",
        flush=True
    )

    return "prepared"


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--raw-root", type=Path, default=RAW_ROOT
    )
    parser.add_argument(
        "--limit", type=int, default=10
    )
    parser.add_argument(
        "--seed", type=int, default=42
    )
    parser.add_argument(
        "--min-pairs", type=int, default=6
    )
    parser.add_argument(
        "--min-ratio", type=float, default=0.20
    )
    parser.add_argument(
        "--frame-gap", type=int, default=10
    )

    args = parser.parse_args()

    data = TartanAir(args.raw_root)

    with open(
        args.raw_root / "depth_statistics.csv"
    ) as f:
        rows = list(csv.DictReader(f))

    rows = [
        r for r in rows
        if float(r["ratio_2m"]) >= args.min_ratio
    ]

    rows.sort(
        key=lambda r: (
            float(r["ratio_1m"]),
            float(r["ratio_2m"])
        ),
        reverse=True
    )

    selected = {}
    counts = Counter()
    completed = 0

    print("Candidate frames:", len(rows))

    for row in rows:

        if completed >= args.limit:
            break

        trajectory = row["trajectory"]
        frame = int(row["frame"])

        previous = selected.setdefault(
            trajectory, []
        )

        if any(
            abs(frame - f) < args.frame_gap
            for f in previous
        ):
            counts["duplicate_neighborhood"] += 1
            continue

        status = prepare_scene(
            data, trajectory, frame, args
        )

        counts[status] += 1

        if status in {"prepared", "existing"}:
            previous.append(frame)
            completed += 1
        else:
            print(
                f"Skip {trajectory}/{frame:06d}: {status}",
                flush=True
            )

    print("\nPreparation summary:")
    for key, value in counts.items():
        print(f"  {key}: {value}")

    print("Total selected groups:", completed)


if __name__ == "__main__":
    main()
