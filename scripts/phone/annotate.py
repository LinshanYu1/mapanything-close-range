"""Click A then B on the model input. Original image pixels are never modified."""

import argparse, math

import numpy as np
from PIL import Image, ImageDraw

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from close_range.common import ROOT, read, save, resolve, sha, phone_distance


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--manifest",
        default="data/manifests/phone/scene-001.json",
    )
    p.add_argument("--points", nargs=4, type=float)
    p.add_argument("--overwrite", action="store_true")
    a = p.parse_args()

    m = read(resolve(a.manifest))
    ref = m["views"][m["reference_index"]]
    path = resolve(ref["image"])
    target = resolve(m["annotation"])
    distance, distance_config = phone_distance(m["scene_id"], m.get("distance_config"))
    print(f"Reference measurement: {distance:g} inch ({distance * 2.54:g} cm)")

    if target.exists() and not a.overwrite:
        p.error("Annotation exists. Use --overwrite deliberately to replace it")
    if sha(path) != ref["sha256"]:
        p.error("Prepared image changed; prepare a new scene")

    im = Image.open(path).convert("RGB")
    if a.points:
        points = np.array(a.points).reshape(2, 2)
    else:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(13, 9))
        ax.imshow(im)
        ax.set_title(
            f"{m['scene_id']}: {distance:g} inch — "
            "Click measured endpoint A, then B. Right click to undo."
        )
        points = np.asarray(plt.ginput(2, timeout=0))
        plt.close(fig)

    if points.shape != (2, 2) or not np.isfinite(points).all():
        p.error("Exactly two finite points required")
    if not (
        (points >= 0).all()
        and (points[:, 0] <= im.width - 1).all()
        and (points[:, 1] <= im.height - 1).all()
    ):
        p.error("Points outside input image")
    if np.linalg.norm(points[0] - points[1]) < 1:
        p.error("Choose two distinct points")

    save(
        target,
        dict(
            scene_id=m["scene_id"],
            reference_view_id=ref["id"],
            image_sha256=ref["sha256"],
            coordinate_space="model_input_pixels_xy",
            pairs=[
                dict(
                    id="pair_001",
                    points_xy=points.tolist(),
                    distance_m=distance * 0.0254,
                    measurement_value=distance,
                    measurement_unit="inch",
                    distance_config=str(distance_config.relative_to(ROOT)),
                    label_source="manual_tape",
                    label_valid=True,
                    measurement_note=(
                        "User-reported measurement; uncertainty not quantified"
                    ),
                )
            ],
        ),
    )

    draw = ImageDraw.Draw(im)
    for label, (x, y) in zip("AB", points):
        draw.ellipse(
            (x - 4, y - 4, x + 4, y + 4),
            outline="red",
            width=2,
        )
        draw.text((x + 7, y - 10), label, fill="red")

    preview = target.with_suffix(".preview.png")
    im.save(preview)
    print(f"Saved {target}\nCheck endpoints in {preview} before running the model.")


if __name__ == "__main__":
    main()
