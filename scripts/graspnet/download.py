"""Download official training archives and extract selected RGB-D scenes."""

import argparse
import shutil
import re
import sys
import zipfile
from pathlib import Path, PurePosixPath

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from close_range.common import ROOT

PACKS = {
    "train_1": "1wQx8IJ_Lok3hVK_nchQUzgw88QBGZ5iq",
    "train_2": "1b1Z1goPV0o_wdwXZ8qTlHd2TBRU5-CmH",
    "train_3": "1oNcmZno2ymsDUWTmfFOxewMBjTXhL95c",
    "train_4": "1e8Xy7-lFhiXk0ugPOKvHKDiGTparmx00",
}


def extract(archive, output, scenes, camera):
    """Select known data members only; never trust archive paths."""
    selected = []
    with zipfile.ZipFile(archive) as zf:
        for info in zf.infolist():
            parts = PurePosixPath(info.filename).parts
            if info.is_dir() or ".." in parts:
                continue
            indexes = [
                i
                for i, part in enumerate(parts)
                if re.fullmatch(r"scene_\d{4}", part)
                and (scenes is None or part in scenes)
            ]
            if not indexes:
                continue
            tail = parts[indexes[0] :]
            if len(tail) < 3 or tail[1] != camera:
                continue
            keep = (
                len(tail) == 4
                and tail[2] in {"rgb", "depth", "label", "meta"}
                and Path(tail[3]).suffix in {".png", ".mat"}
            ) or (
                len(tail) == 3
                and tail[2] in {"camK.npy", "camera_poses.npy", "cam0_wrt_table.npy"}
            )
            if keep:
                selected.append((info, output / "scenes" / Path(*tail)))
        found = {p.relative_to(output / "scenes").parts[0] for _, p in selected}
        if scenes is not None and scenes - found:
            raise ValueError(
                f"Requested scenes absent from archive: {sorted(scenes - found)}"
            )
        if not selected:
            raise ValueError(
                "No requested camera/scenes found in this archive; check the selected pack."
            )
        for info, target in selected:
            if target.exists():
                raise FileExistsError(f"Refusing to overwrite: {target}")
        for info, target in selected:
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_suffix(target.suffix + ".part")
            with zf.open(info) as source, temporary.open("wb") as dest:
                shutil.copyfileobj(source, dest)
            temporary.replace(target)
    print(f'Extracted {len(selected)} files to {output / "scenes"}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack", choices=PACKS, default="train_1")
    parser.add_argument(
        "--archive", type=Path, help="Use an existing ZIP; skip download"
    )
    parser.add_argument(
        "--scenes",
        nargs="+",
        help="Only these scenes; omit to extract all scenes in the ZIP",
    )
    parser.add_argument(
        "--camera", choices=["realsense", "kinect"], default="realsense"
    )
    parser.add_argument("--output", type=Path, default=ROOT / "data/raw/graspnet")
    args = parser.parse_args()
    if any(
        len(s) != 10 or not s.startswith("scene_") or not s[6:].isdigit()
        for s in (args.scenes or [])
    ):
        parser.error("Scene names must look like scene_0017")
    archive = args.archive
    if archive is None:
        archive = args.output / "archives" / f"{args.pack}.zip"
        if not archive.exists():
            try:
                import gdown
            except ImportError:
                parser.error(
                    "Install gdown in the Python environment, or pass --archive downloaded.zip"
                )
            archive.parent.mkdir(parents=True, exist_ok=True)
            partial = archive.with_suffix(".zip.part")
            print(
                "Downloading the FULL official pack (6.3–20 GB), not only selected scenes.",
                flush=True,
            )
            result = gdown.download(
                id=PACKS[args.pack], output=str(partial), resume=True
            )
            if result is None or not zipfile.is_zipfile(partial):
                raise RuntimeError(
                    "Download failed or did not return a ZIP. Use the official website alternative mirrors."
                )
            partial.replace(archive)
    extract(
        archive, args.output, set(args.scenes) if args.scenes else None, args.camera
    )


if __name__ == "__main__":
    main()
