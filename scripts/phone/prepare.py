"""Prepare immutable RGB model inputs with the pinned upstream crop/resize."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from close_range.common import ROOT, save, sha, phone_distance
from close_range.phone_checks import check_orientation, scene_images
from PIL import Image, ImageOps
from mapanything.utils.image import find_closest_aspect_ratio
from mapanything.utils.cropping import crop_resize_if_necessary


def prepare_scene(a):
    if "/" in a.scene or ".." in a.scene:
        raise ValueError("Use a simple scene name")
    distance, distance_config = phone_distance(a.scene)
    raw = ROOT / "data/raw/phone" / a.scene
    out = ROOT / "data/processed/phone" / a.scene
    manifest = ROOT / "data/manifests/phone" / f"{a.scene}.json"
    if manifest.is_file() and out.is_dir():
        print(f"[{a.scene}] skipped existing prepared scene")
        return
    if manifest.exists() or out.exists():
        raise ValueError(
            "Prepared scene already exists; use a new scene name to avoid invalidating annotations"
        )
    files = scene_images(raw)
    imgs = []
    for f in files:
        with Image.open(f) as im:
            imgs.append(ImageOps.exif_transpose(im).convert("RGB"))
    kind = check_orientation(
        [f.name for f in files],
        [im.size for im in imgs],
        getattr(a, "allow_mixed_orientation", False),
    )
    size = find_closest_aspect_ratio(
        sum(im.width / im.height for im in imgs) / len(imgs), 518
    )
    out.mkdir(parents=True)
    views = []
    for i, (f, im) in enumerate(zip(files, imgs)):
        dest = out / f"view_{i+1:03d}.png"
        crop_resize_if_necessary(im, resolution=size)[0].save(dest)
        views.append(
            dict(
                id=f"view_{i+1:03d}",
                image=str(dest.relative_to(ROOT)),
                sha256=sha(dest),
                original=str(f.relative_to(ROOT)),
                oriented_original_size=list(im.size),
            )
        )
    save(
        manifest,
        dict(
            schema_version=1,
            scene_id=a.scene,
            source="phone",
            distance_config=str(distance_config.relative_to(ROOT)),
            views=views,
            reference_index=0,
            model_size=list(size),
            annotation=f"data/annotations/phone/{a.scene}.json",
            preprocessing="Upstream crop_resize_if_necessary, EXIF oriented RGB, annotate model-input pixels",
        ),
    )
    print(
        f"[{a.scene}] Prepared {len(views)} {kind} images at {size}. "
        f"Reference: {files[0].name}. Manifest: {manifest}"
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scene", help="Only this scene; omit to scan all phone scenes")
    p.add_argument(
        "--allow-mixed-orientation",
        action="store_true",
        help="Accept landscape and portrait photos in one scene (endpoints may be cropped)",
    )
    a = p.parse_args()
    from close_range.scene_batch import run_scenes

    run_scenes(
        ROOT / "data/raw/phone",
        a.scene,
        lambda scene: prepare_scene(
            argparse.Namespace(
                scene=scene, allow_mixed_orientation=a.allow_mixed_orientation
            )
        ),
    )


if __name__ == "__main__":
    main()
