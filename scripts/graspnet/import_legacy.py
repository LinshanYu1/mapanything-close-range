"""Import an existing GraspNet pilot manifest into the shared inference format."""

import argparse, shutil
from pathlib import Path
from PIL import Image
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from close_range.common import ROOT, read, save, sha


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--legacy-manifest", type=Path, required=True)
    p.add_argument("--legacy-root", type=Path, required=True)
    p.add_argument("--scene", default="graspnet_pilot")
    a = p.parse_args()
    if "/" in a.scene or ".." in a.scene:
        p.error("Use a simple scene name")
    old = read(a.legacy_manifest)
    eligible = [i for i, r in enumerate(old["frames"]) if r["qc_pass"]]
    if not eligible:
        p.error("No valid reference labels")
    ref = eligible[0]
    out = ROOT / "data/processed/graspnet" / a.scene
    if out.exists():
        p.error("Scene exists; use a different name")
    out.mkdir(parents=True)
    views = []
    size = None
    for i, r in enumerate(old["frames"]):
        src = a.legacy_root / r["input_image"]
        dest = out / f"view_{i+1:03d}.png"
        shutil.copy2(src, dest)
        with Image.open(dest) as im:
            if size is not None and im.size != size:
                raise ValueError("Mixed model input dimensions")
            size = im.size
        views.append(
            dict(
                id=f"view_{i+1:03d}",
                image=str(dest.relative_to(ROOT)),
                sha256=sha(dest),
                legacy_frame=r["frame"],
            )
        )
    annotation = f"data/annotations/graspnet/{a.scene}.json"
    save(
        ROOT / annotation,
        dict(
            scene_id=a.scene,
            reference_view_id=views[ref]["id"],
            image_sha256=views[ref]["sha256"],
            coordinate_space="model_input_pixels_xy",
            pairs=[
                dict(
                    id="pair_001",
                    points_xy=old["frames"][ref]["points_model"],
                    distance_m=old["reference_distance_m"],
                    label_source="sensor_depth",
                    label_valid=True,
                    measurement_note=old.get("reference", "Legacy sensor reference"),
                )
            ],
        ),
    )
    save(
        ROOT / "data/manifests/graspnet" / f"{a.scene}.json",
        dict(
            schema_version=1,
            scene_id=a.scene,
            source="graspnet",
            views=views,
            reference_index=ref,
            model_size=list(size),
            annotation=annotation,
            preprocessing=old["preprocessing"],
            legacy_manifest=str(a.legacy_manifest.resolve()),
        ),
    )
    save(out / "legacy_manifest.json", old)
    print(
        "Imported reference pair plus all views. Legacy QC retained. Uses shared run_baseline.py and evaluate.py."
    )


if __name__ == "__main__":
    main()
