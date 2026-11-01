"""Prepare a GraspNet view group and sensor-derived reference point-pair labels."""

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.io import loadmat
from scipy.ndimage import binary_erosion

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from close_range.common import ROOT, save, sha
from mapanything.utils.cropping import crop_resize_if_necessary
from mapanything.utils.image import find_closest_aspect_ratio


def backproject(points, depth, intrinsic):
    points = np.asarray(points, dtype=int)
    z = depth[points[:, 1], points[:, 0]]
    rays = np.linalg.solve(intrinsic, np.c_[points, np.ones(len(points))].T).T
    return rays * z[:, None]


def prepare_scene(args):
    name = args.name or f"{args.scene}_{args.camera}_pilot"
    if any("/" in x or "\\" in x or ".." in x for x in [args.scene, name]):
        raise ValueError("Use simple scene/group names")
    if args.pairs < 1 or any(
        not f.isdigit() or not 0 <= int(f) <= 255 for f in args.frames
    ):
        raise ValueError("Positive pair count and frame IDs 0000–0255 required")
    frames = [f"{int(f):04d}" for f in args.frames]
    if len(set(frames)) != len(frames):
        raise ValueError("Duplicate frames are not allowed")
    raw = args.raw_root.resolve() / args.scene / args.camera
    data = args.data_root.resolve()
    out = data / "processed/graspnet" / name
    ann = data / "annotations/graspnet" / f"{name}.json"
    manifest = data / "manifests/graspnet" / f"{name}.json"
    if out.is_dir() and ann.is_file() and manifest.is_file():
        print(f"[{args.scene}] skipped existing group: {name}")
        return
    if any(p.exists() for p in [out, ann, manifest]):
        raise ValueError("Partial output exists; inspect it and use a new --name")
    # Check every selected RGB-D record before writing outputs.
    for frame in frames:
        for folder, extension in [
            ("rgb", "png"),
            ("depth", "png"),
            ("label", "png"),
            ("meta", "mat"),
        ]:
            if not (raw / folder / f"{frame}.{extension}").is_file():
                raise ValueError(f"Missing {folder}/{frame}.{extension}")
    reference = frames[0]
    meta = loadmat(raw / "meta" / f"{reference}.mat")
    intrinsic = np.asarray(meta["intrinsic_matrix"], dtype=float)
    factor = float(np.asarray(meta["factor_depth"]).squeeze())
    if (
        not np.isfinite(factor)
        or factor <= 0
        or intrinsic.shape != (3, 3)
        or not np.isfinite(intrinsic).all()
    ):
        raise ValueError("Invalid depth factor or camera intrinsics")
    depth = (
        np.asarray(Image.open(raw / "depth" / f"{reference}.png"), dtype=float) / factor
    )
    labels = np.asarray(Image.open(raw / "label" / f"{reference}.png"))
    image = Image.open(raw / "rgb" / f"{reference}.png").convert("RGB")
    if depth.shape != (image.height, image.width) or labels.shape != depth.shape:
        raise ValueError("RGB/depth/label sizes differ")
    size = find_closest_aspect_ratio(image.width / image.height, 518)
    # Transform an integer source-pixel map with the exact RGB crop and nearest sampling.
    source_ids = np.arange(depth.size, dtype=np.int32).reshape(depth.shape)
    processed, quantities = crop_resize_if_necessary(
        image, resolution=size, additional_quantities=[source_ids]
    )
    ids = np.asarray(quantities[0], dtype=np.int64)
    source_y, source_x = np.divmod(ids, image.width)
    valid = np.isfinite(depth) & (depth > 0)
    interior = np.zeros_like(valid)
    for label in np.unique(labels):
        if label > 0:
            interior |= binary_erosion((labels == label) & valid, iterations=3)
    eligible = interior[source_y, source_x]
    candidates = np.argwhere(eligible)
    rng = np.random.default_rng(args.seed)
    pairs = []
    used = set()
    for _ in range(max(1000, args.pairs * 1000)):
        if len(candidates) < 2 or len(pairs) == args.pairs:
            break
        pixels = candidates[rng.choice(len(candidates), 2, replace=False)][:, ::-1]
        x, y = pixels.T
        original = np.c_[source_x[y, x], source_y[y, x]]
        ox, oy = original.T
        if labels[oy[0], ox[0]] == labels[oy[1], ox[1]]:
            continue
        key = tuple(sorted(map(tuple, pixels.tolist())))
        if key in used:
            continue
        xyz = backproject(original, depth, intrinsic)
        distance = float(np.linalg.norm(xyz[0] - xyz[1]))
        if not np.isfinite(distance) or distance <= 0:
            continue
        used.add(key)
        pairs.append(
            dict(
                id=f"pair_{len(pairs)+1:03d}",
                points_xy=pixels.tolist(),
                source_points_xy=original.tolist(),
                distance_m=distance,
                label_source="sensor_depth",
                label_valid=True,
            )
        )
    if len(pairs) != args.pairs:
        raise ValueError(
            f"Only {len(pairs)} valid distinct-object pairs available; reduce --pairs"
        )
    out.mkdir(parents=True)
    views = []
    for i, frame in enumerate(frames):
        path = raw / "rgb" / f"{frame}.png"
        with Image.open(path) as img:
            img = img.convert("RGB")
            if img.size != image.size:
                raise ValueError("Mixed image dimensions")
            dest = out / f"view_{i+1:03d}.png"
            crop_resize_if_necessary(img, resolution=size)[0].save(dest)
        views.append(
            dict(
                id=f"view_{i+1:03d}",
                image=str(dest),
                sha256=sha(dest),
                original=str(path),
                frame=frame,
            )
        )
    save(
        ann,
        dict(
            scene_id=name,
            reference_view_id=views[0]["id"],
            image_sha256=views[0]["sha256"],
            coordinate_space="model_input_pixels_xy",
            pairs=pairs,
            sensor_reference=dict(
                depth=str(raw / "depth" / f"{reference}.png"),
                meta=str(raw / "meta" / f"{reference}.mat"),
                factor_depth=factor,
            ),
            note="Sensor-derived labels, not independent tape measurements. Interior mask is only a basic quality filter.",
        ),
    )
    save(
        manifest,
        dict(
            schema_version=1,
            scene_id=name,
            source="graspnet",
            original_scene=args.scene,
            camera=args.camera,
            views=views,
            reference_index=0,
            model_size=list(size),
            annotation=str(ann),
            preprocessing="Upstream crop/resize; nearest source-pixel mapping for sensor labels",
            seed=args.seed,
        ),
    )
    previews = ann.parent / "previews" / name
    previews.mkdir(parents=True)
    for pair in pairs:
        preview = processed.copy()
        draw = ImageDraw.Draw(preview)
        for letter, (x, y) in zip("AB", pair["points_xy"]):
            draw.ellipse((x - 4, y - 4, x + 4, y + 4), outline="red", width=2)
            draw.text((x + 6, y), letter, fill="red")
        preview.save(previews / f'{pair["id"]}.png')
    print(f"Prepared {len(views)} views and {len(pairs)} pairs. Manifest: {manifest}")
    print(f"Inspect point-pair previews: {previews}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-root", type=Path, default=ROOT / "data/raw/graspnet/scenes"
    )
    parser.add_argument("--scene", help="Only this scene; omit to scan raw-root")
    parser.add_argument(
        "--camera", choices=["realsense", "kinect"], default="realsense"
    )
    parser.add_argument("--frames", nargs="+", default=["0000", "0012", "0024"])
    parser.add_argument("--name", help="Unique output group name")
    parser.add_argument("--pairs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--data-root", type=Path, default=ROOT / "data", help="Output data root"
    )
    args = parser.parse_args()
    if args.name and not args.scene:
        parser.error("--name requires --scene")
    from close_range.scene_batch import run_scenes

    run_scenes(
        args.raw_root,
        args.scene,
        lambda scene: prepare_scene(
            argparse.Namespace(**{**vars(args), "scene": scene})
        ),
    )


if __name__ == "__main__":
    main()
