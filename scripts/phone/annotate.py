"""Click A then B on the model input. Original image pixels are never modified.

Without --scene/--manifest, every prepared phone scene that has no annotation
yet is opened in turn. Close a window without clicking two points to stop.
"""

import argparse

import numpy as np
from PIL import Image, ImageDraw

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from close_range.common import ROOT, read, save, resolve, sha, phone_distance

MANIFESTS = "data/manifests/phone"


class Stop(Exception):
    """The user closed the window before choosing both points."""


def scene_manifest(scene, root=None):
    if not scene or scene in {".", ".."} or "/" in scene or "\\" in scene:
        raise ValueError("Use a simple scene name, e.g. scene-002")
    path = Path(MANIFESTS) / f"{scene}.json"
    if not (Path(root) if root else ROOT).joinpath(path).is_file():
        raise ValueError(
            f"No manifest {path}; run scripts/phone/prepare.py --scene {scene} first"
        )
    return path


def pending_manifests(root=None):
    """Prepared scenes (manifest + reference image present) without annotation."""
    root = Path(root) if root else ROOT
    pending, skipped = [], []
    for path in sorted((root / MANIFESTS).glob("*.json")):
        m = read(path)
        ref = root / m["views"][m["reference_index"]]["image"]
        if not ref.is_file():
            skipped.append((m["scene_id"], "reference image missing"))
        elif (root / m["annotation"]).exists():
            skipped.append((m["scene_id"], "already annotated"))
        else:
            pending.append(path.relative_to(root))
    return pending, skipped


def click_points(im, title):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(13, 9))
    ax.imshow(im)
    ax.set_title(title)
    points = np.asarray(plt.ginput(2, timeout=0))
    plt.close(fig)
    if len(points) < 2:
        raise Stop
    return points


def annotate(manifest, overwrite=False, points=None, position=""):
    m = read(resolve(manifest))
    ref = m["views"][m["reference_index"]]
    path = resolve(ref["image"])
    target = resolve(m["annotation"])
    distance, distance_config = phone_distance(m["scene_id"], m.get("distance_config"))
    print(
        f"[{m['scene_id']}]{position} Reference measurement: "
        f"{distance:g} inch ({distance * 2.54:g} cm)"
    )

    if target.exists() and not overwrite:
        raise ValueError("Annotation exists. Use --overwrite deliberately to replace it")
    if sha(path) != ref["sha256"]:
        raise ValueError("Prepared image changed; prepare a new scene")

    im = Image.open(path).convert("RGB")
    if points is None:
        points = click_points(
            im,
            f"{m['scene_id']}{position}: {distance:g} inch ({distance * 2.54:g} cm) — "
            "Click measured endpoint A, then B. Right click to undo. "
            "Close window to stop.",
        )
    points = np.asarray(points, dtype=float).reshape(-1, 2)

    if points.shape != (2, 2) or not np.isfinite(points).all():
        raise ValueError("Exactly two finite points required")
    if not (
        (points >= 0).all()
        and (points[:, 0] <= im.width - 1).all()
        and (points[:, 1] <= im.height - 1).all()
    ):
        raise ValueError("Points outside input image")
    if np.linalg.norm(points[0] - points[1]) < 1:
        raise ValueError("Choose two distinct points")

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


def main():
    p = argparse.ArgumentParser(description=__doc__)
    which = p.add_mutually_exclusive_group()
    which.add_argument("--scene", help="Annotate this scene, e.g. scene-002")
    which.add_argument("--manifest", help="Annotate the scene of this manifest")
    p.add_argument("--points", nargs=4, type=float, metavar=("AX", "AY", "BX", "BY"))
    p.add_argument("--overwrite", action="store_true")
    a = p.parse_args()

    if a.scene or a.manifest:
        try:
            manifest = scene_manifest(a.scene) if a.scene else Path(a.manifest)
            annotate(manifest, a.overwrite, a.points)
        except Stop:
            p.error("Window closed before two points were chosen; nothing saved")
        except ValueError as error:
            p.error(str(error))
        return

    if a.overwrite or a.points:
        p.error("--overwrite and --points need --scene or --manifest")
    pending, skipped = pending_manifests()
    for scene, reason in skipped:
        print(f"[{scene}] skipped: {reason}")
    if not pending:
        print("Nothing to annotate.")
        return
    done, failed = 0, []
    for i, manifest in enumerate(pending, 1):
        try:
            annotate(manifest, position=f" ({i}/{len(pending)})")
            done += 1
        except Stop:
            print("Window closed; stopping. Rerun to continue with the rest.")
            break
        except ValueError as error:
            print(f"[{manifest.stem}] FAILED: {error}")
            failed.append(manifest.stem)
    remaining = len(pending) - done - len(failed)
    print(f"Annotated: {done}; failed: {len(failed)}; remaining: {remaining}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
