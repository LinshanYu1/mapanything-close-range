"""Sequential scene discovery with per-scene error reporting."""

from pathlib import Path


def run_scenes(raw_root, scene, prepare):
    raw_root = Path(raw_root)
    if scene is not None:
        if not scene or scene in {".", ".."} or "/" in scene or "\\" in scene:
            raise ValueError("Use a simple scene directory name")
        scenes = [scene]
    else:
        scenes = sorted(
            p.name
            for p in raw_root.iterdir()
            if p.is_dir() and not p.name.startswith(".")
        )
    if not scenes:
        raise ValueError(f"No scene directories in {raw_root}")
    failed = []
    for name in scenes:
        try:
            prepare(name)
        except Exception as error:
            print(f"[{name}] FAILED: {error}", flush=True)
            failed.append(name)
    print(f"Scenes scanned: {len(scenes)}; failed: {len(failed)}")
    if failed:
        raise SystemExit(1)
