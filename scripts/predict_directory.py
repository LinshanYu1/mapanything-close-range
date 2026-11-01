"""One immediate child directory = one scene; automatically infer and score each scene."""

import argparse
import csv
from datetime import datetime
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from close_range.common import ROOT, read, save, resolve, sha
from run_baseline import load_model, prepare_inputs, predict_scene
from evaluate import evaluate_run

EXTENSIONS = {".png", ".jpg", ".jpeg"}


def scene_manifest(scene, manifests):
    """Use directory contents, while checking any existing manifest matches exactly."""
    from PIL import Image

    image_dir = scene / "images" if (scene / "images").is_dir() else scene
    if image_dir != scene and any(
        p.suffix.lower() in EXTENSIONS for p in scene.iterdir() if p.is_file()
    ):
        raise ValueError(
            "Images found both at scene root and in images/: choose one layout"
        )
    files = sorted(
        p
        for p in image_dir.iterdir()
        if p.is_file()
        and p.suffix.lower() in EXTENSIONS
        and not p.name.startswith(".")
        and ".preview." not in p.name.lower()
    )
    if not files:
        raise ValueError("No input JPG/PNG images in this scene")
    manifest_path = manifests / f"{scene.name}.json"
    if manifest_path.exists():
        manifest = read(manifest_path)
        listed = [resolve(v["image"]).resolve() for v in manifest["views"]]
        if len(listed) != len(files) or set(listed) != {p.resolve() for p in files}:
            raise ValueError(
                "Directory images differ from manifest; prepare/update this scene first"
            )
        if manifest["scene_id"] != scene.name:
            raise ValueError("Scene ID differs from directory name")
        return manifest, manifest_path
    # No manifest: prediction is possible, but coordinates/labels cannot be assumed.
    sizes = []
    for path in files:
        with Image.open(path) as im:
            if im.getexif().get(274, 1) != 1:
                raise ValueError(
                    "Input orientation is not normalized; preprocess first"
                )
            sizes.append(im.size)
    if len(set(sizes)) != 1 or any(n % 14 for n in sizes[0]):
        raise ValueError(
            "Expected preprocessed equal-sized images with dimensions divisible by 14"
        )
    return (
        dict(
            schema_version=1,
            scene_id=scene.name,
            source="unlabeled",
            views=[
                dict(id=p.stem, image=str(p.resolve()), sha256=sha(p)) for p in files
            ],
            reference_index=0,
            model_size=list(sizes[0]),
        ),
        None,
    )


def write_summary(out, records):
    save(
        out / "summary.json",
        dict(
            scenes=records,
            scene_count=len(records),
            evaluated=sum(r["status"] == "evaluated" for r in records),
            prediction_only=sum(
                r["status"] == "predicted_without_labels" for r in records
            ),
            failed=sum(r["status"] in {"failed", "evaluation_failed"} for r in records),
        ),
    )
    fields = [
        "scene_id",
        "image_count",
        "mode",
        "status",
        "error",
        "output",
        "pair_id",
        "prediction_valid",
        "reference_inch",
        "predicted_inch",
        "reference_cm",
        "predicted_cm",
        "absolute_error_cm",
        "relative_error_pct",
        "success_10pct",
    ]
    with (out / "summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for record in records:
            for result in record.get("results") or [{}]:
                writer.writerow({**record, **result})


def run_batch(input_dir, manifests, out, device, check_only=False):
    scenes = sorted(
        p for p in input_dir.iterdir() if p.is_dir() and not p.name.startswith(".")
    )
    if not scenes:
        raise ValueError(
            "Input must be a parent directory containing scene subdirectories"
        )
    out.mkdir(parents=True, exist_ok=False)
    records, model = [], None
    for scene in scenes:
        record = dict(
            scene_id=scene.name,
            image_count=0,
            mode="",
            status="failed",
            error="",
            output=str(out / scene.name),
        )
        try:
            manifest, manifest_path = scene_manifest(scene, manifests)
            count = len(manifest["views"])
            mode = "single" if count == 1 else "multi"
            record.update(image_count=count, mode=mode)
            print(f"[{scene.name}] {count} image(s) -> {mode}", flush=True)
            prepared = prepare_inputs(manifest, mode)
            if check_only:
                record["status"] = "checked"
            else:
                if model is None:
                    model = load_model(device)
                scene_out = out / scene.name
                predict_scene(model, manifest, mode, device, scene_out, prepared)
                save(scene_out / "input_manifest.json", manifest)
                annotation = manifest.get("annotation")
                if annotation and resolve(annotation).is_file():
                    try:
                        summary = evaluate_run(manifest, scene_out)
                        record.update(status="evaluated", results=summary["results"])
                    except Exception as exc:
                        record.update(status="evaluation_failed", error=str(exc))
                else:
                    record.update(
                        status="predicted_without_labels",
                        error="Prediction saved; no annotation, so distance was not evaluated",
                    )
        except Exception as exc:
            record.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        records.append(record)
        write_summary(out, records)
        print(f"[{scene.name}] {record['status']} {record['error']}", flush=True)
    print(f'Batch summary: {out / "summary.csv"}', flush=True)
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", required=True, help="Parent directory of preprocessed scene folders"
    )
    parser.add_argument(
        "--manifests-dir", help="Default: data/manifests/<input directory name>"
    )
    parser.add_argument("--output", help="Default: outputs/batch_<timestamp>")
    parser.add_argument("--device", choices=["mps", "cuda", "cpu"], default="mps")
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Validate scenes without loading model",
    )
    args = parser.parse_args()
    input_dir = resolve(args.input).resolve()
    if not input_dir.is_dir():
        parser.error("Input directory does not exist")
    manifests = (
        resolve(args.manifests_dir)
        if args.manifests_dir
        else ROOT / "data/manifests" / input_dir.name
    )
    out = (
        resolve(args.output)
        if args.output
        else ROOT / "outputs" / ("batch_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    )
    if out.resolve().is_relative_to(input_dir):
        parser.error("Output must be outside the input directory")
    records = run_batch(input_dir, manifests, out, args.device, args.check_only)
    if any(r["status"] in {"failed", "evaluation_failed"} for r in records):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
