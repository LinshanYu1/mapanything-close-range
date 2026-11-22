"""Fetch shared phone photos into data/raw/phone, then run prepare.py as usual.

The shared folder holds scene-XXX subfolders and distances.conf, e.g. a Google
Drive folder shared as "Anyone with the link". Existing local scenes and
distance entries are never overwritten.

  download.py --folder https://drive.google.com/drive/folders/<ID>
  download.py --source ~/Downloads/phone_zongyu            # already downloaded
  download.py --source ~/Downloads/phone_zongyu-001.zip    # Drive "Download" zip
"""

import argparse
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from close_range.common import ROOT, phone_distance
from close_range.phone_import import (
    fetch_drive,
    find_root,
    import_scenes,
    merge_distances,
    safe_extract,
)


def collect(sources, work):
    """Resolve each source (folder or zip) to a folder holding scene-XXX folders."""
    roots = []
    for i, src in enumerate(sources):
        src = Path(src).expanduser()
        if src.is_file() and src.suffix.lower() == ".zip":
            src = safe_extract(src, work / f"zip{i}")
        elif not src.is_dir():
            raise ValueError(f"Not a folder or .zip: {src}")
        roots.append(find_root(src))
    return roots


def report(title, result):
    for kind in ("added", "skipped", "repaired"):
        if result.get(kind):
            print(f"{title} {kind}: {', '.join(result[kind])}")
    for item in result["conflicts"]:
        print(f"{title} CONFLICT: {item}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    which = p.add_mutually_exclusive_group(required=True)
    which.add_argument("--folder", help="Google Drive folder URL or ID")
    which.add_argument("--source", nargs="+", help="Downloaded folder(s) or .zip file(s)")
    p.add_argument("--dest", default="data/raw/phone", help="Default: data/raw/phone")
    a = p.parse_args()

    dest = (ROOT / a.dest).resolve()
    conflicts = 0
    with tempfile.TemporaryDirectory(prefix="phone-download-") as tmp:
        work = Path(tmp)
        try:
            if a.folder:
                drive = fetch_drive(a.folder, work / "drive", dest)
                if drive["present"]:
                    print("Reused valid local files in scenes: " + ", ".join(drive["present"]))
                for path in drive["duplicates"]:
                    print(f"WARNING duplicate file name on Drive, used the first copy: {path}")
                for scene in drive["failed"]:
                    print(f"FAILED {scene}: Drive refused some files (rate limit?); rerun later")
                conflicts += len(drive["failed"])
                got = work / "drive"
                roots = [find_root(got)] if got.is_dir() and any(got.rglob("*")) else []
            else:
                roots = collect(a.source, work)
        except Exception as error:
            if a.folder and not isinstance(error, ValueError):
                p.error(
                    f"Google Drive download failed ({type(error).__name__}: {error}).\n"
                    "Check the folder is shared as 'Anyone with the link'. If Drive is "
                    "rate-limiting, rerun later, or download it in a browser and use "
                    "--source <downloaded folder or zip>."
                )
            p.error(str(error))

        added_scenes = []
        for root in roots:
            print(f"Importing from {root}")
            scenes = import_scenes(root, dest)
            report("Scenes", scenes)
            added_scenes += scenes["added"] + scenes["repaired"]
            conflicts += len(scenes["conflicts"])
            conf = root / "distances.conf"
            if conf.is_file():
                dist = merge_distances(conf, dest / "distances.conf")
                report("Distances", dist)
                conflicts += len(dist["conflicts"])
            else:
                print("No distances.conf in the shared folder")

    conf_path = dest / "distances.conf"
    for scene in added_scenes:
        try:
            phone_distance(scene, conf_path)
        except (ValueError, OSError) as error:
            print(f"WARNING {scene}: {error}")
    print(f"Done. Next: bash scripts/local.sh scripts/phone/prepare.py")
    if conflicts:
        print(f"{conflicts} problem(s) need a manual check; nothing was overwritten.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
