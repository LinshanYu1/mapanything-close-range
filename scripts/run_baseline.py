"""RGB-only inference helpers and the optional single-scene command."""

import argparse
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from close_range.common import ROOT, read, save, resolve, sha, REVISION


def load_model(device):
    import torch
    from mapanything.models import MapAnything

    if device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError(
            "MPS unavailable. Run in a normal Mac terminal, or explicitly select another device."
        )
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    print("Loading pinned pretrained model...", flush=True)
    return (
        MapAnything.from_pretrained("facebook/map-anything", revision=REVISION)
        .to(device)
        .eval()
    )


def prepare_inputs(manifest, mode):
    from PIL import Image
    from mapanything.utils.image import load_images

    ref = manifest["reference_index"]
    indices = (
        [ref]
        if mode == "single"
        else [ref] + [i for i in range(len(manifest["views"])) if i != ref]
    )
    chosen = [manifest["views"][i] for i in indices]
    size = tuple(manifest["model_size"])
    if len(size) != 2 or any(n <= 0 or n % 14 for n in size):
        raise ValueError("Prepared image dimensions must be positive multiples of 14")
    files = []
    for view in chosen:
        path = resolve(view["image"])
        if sha(path) != view["sha256"]:
            raise ValueError(f"Input changed since preparation: {path}")
        with Image.open(path) as im:
            if im.size != size:
                raise ValueError(f"Unexpected input size: {path}")
        files.append(str(path))
    views = load_images(files, resize_mode="fixed_size", size=size)
    if len(views) != len(files):
        raise ValueError("Loader skipped an image")
    for view in views:
        if tuple(view["img"].shape[-2:]) != size[::-1]:
            raise ValueError("Unexpected image transform")
    return views, chosen, size


def predict_scene(model, manifest, mode, device, out, prepared=None):
    import numpy as np
    import torch

    views, chosen, size = (
        prepared if prepared is not None else prepare_inputs(manifest, mode)
    )
    if out.exists():
        raise FileExistsError(f"Output already exists: {out}")
    start = time.time()
    with torch.inference_mode():
        predictions = model.infer(
            views,
            memory_efficient_inference=True,
            minibatch_size=1,
            use_amp=device == "cuda",
            apply_mask=True,
            mask_edges=True,
            apply_confidence_mask=False,
        )
    xyz = predictions[0]["pts3d"][0].detach().cpu().numpy()
    mask = predictions[0]["mask"][0].detach().cpu().numpy().squeeze()
    if xyz.shape != (*size[::-1], 3) or mask.shape != size[::-1]:
        raise ValueError("Unexpected point map shape")
    out.mkdir(parents=True)
    np.savez_compressed(out / "reference_prediction.npz", pts3d=xyz, mask=mask)
    save(
        out / "run.json",
        dict(
            scene_id=manifest["scene_id"],
            source=manifest["source"],
            mode=mode,
            reference_view_id=chosen[0]["id"],
            reference_image=chosen[0]["image"],
            reference_sha256=chosen[0]["sha256"],
            input_views=chosen,
            model="facebook/map-anything",
            revision=REVISION,
            device=device,
            torch_version=torch.__version__,
            inference_seconds=time.time() - start,
            units="m",
            rgb_only=True,
            upstream=read(ROOT / "third_party/upstream.json"),
        ),
    )
    print(f"Prediction saved: {out}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/manifests/phone/scene-001.json")
    parser.add_argument("--mode", choices=["single", "multi"], default="multi")
    parser.add_argument("--device", choices=["mps", "cuda", "cpu"], default="mps")
    parser.add_argument("--output", required=True)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    manifest = read(resolve(args.manifest))
    prepared = prepare_inputs(manifest, args.mode)
    print(
        f"Input check passed: {len(prepared[0])} RGB images; no labels given to model",
        flush=True,
    )
    if args.check_only:
        return
    out = resolve(args.output)
    if out.exists():
        raise FileExistsError("Choose a new output directory")
    model = load_model(args.device)
    predict_scene(model, manifest, args.mode, args.device, out, prepared)


if __name__ == "__main__":
    main()
