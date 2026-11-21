"""Import shared phone scenes (scene folders + distances.conf) into data/raw/phone.

Nothing already present is overwritten: identical scenes are skipped, and any
difference is reported as a conflict for a person to resolve.
"""

import re
import shutil
import tempfile
import zipfile
from pathlib import Path, PurePosixPath

SCENE_RE = re.compile(r"scene-\d{3,}")
ENTRY_RE = re.compile(r"\s*([^#;=\s][^=]*?)\s*=\s*(.*?)\s*$")


def _files(folder):
    return sorted(
        f for f in Path(folder).iterdir() if f.is_file() and not f.name.startswith(".")
    )


def _scene_dirs(folder):
    return sorted(
        d for d in Path(folder).iterdir() if d.is_dir() and SCENE_RE.fullmatch(d.name)
    )


def find_root(path, max_depth=3):
    """The folder that directly holds scene-XXX folders (Drive adds wrapper folders)."""
    path = Path(path)
    level = [path]
    for _ in range(max_depth + 1):
        hits = [p for p in level if _scene_dirs(p) or (p / "distances.conf").is_file()]
        if len(hits) == 1:
            return hits[0]
        if len(hits) > 1:
            names = ", ".join(str(h) for h in hits)
            raise ValueError(f"Several folders look like phone data: {names}")
        level = [
            d for p in level for d in p.iterdir() if d.is_dir() and not d.name.startswith(".")
        ]
    raise ValueError(f"No scene-XXX folders or distances.conf found under {path}")


def safe_extract(archive, output):
    """Extract a zip without letting member paths escape the output folder."""
    output = Path(output)
    with zipfile.ZipFile(archive) as zf:
        for info in zf.infolist():
            parts = PurePosixPath(info.filename).parts
            if info.filename.startswith("/") or ".." in parts:
                raise ValueError(f"Unsafe path in archive: {info.filename}")
            if parts and parts[0] == "__MACOSX":
                continue
            zf.extract(info, output)
    return output


def import_scenes(src_root, dest):
    """Copy new scene folders; returns dict of added / skipped / conflicts."""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    result = dict(added=[], skipped=[], conflicts=[])
    for scene in _scene_dirs(src_root):
        files = _files(scene)
        target = dest / scene.name
        if target.exists():
            mine = {f.name: f.stat().st_size for f in _files(target)}
            theirs = {f.name: f.stat().st_size for f in files}
            if mine == theirs:
                result["skipped"].append(scene.name)
            else:
                result["conflicts"].append(
                    f"{scene.name}: local folder differs from the shared one; left unchanged"
                )
            continue
        if not files:
            result["conflicts"].append(f"{scene.name}: shared folder is empty; not imported")
            continue
        tmp = Path(tempfile.mkdtemp(prefix=f".{scene.name}-", dir=dest))
        try:
            for f in files:
                shutil.copy2(f, tmp / f.name)
            tmp.rename(target)
        except BaseException:
            shutil.rmtree(tmp, ignore_errors=True)
            raise
        result["added"].append(scene.name)
    return result


def _entries(text):
    """{key: (value, [line indexes incl. preceding comment lines])} in [distances]."""
    out, section, pending = {}, None, []
    for i, line in enumerate(text.splitlines()):
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            section, pending = stripped[1:-1].strip(), []
        elif section == "distances" and stripped.startswith(("#", ";")):
            pending.append(i)
        elif section == "distances" and (m := ENTRY_RE.fullmatch(line)):
            value = re.split(r"\s+[#;]", m.group(2))[0].strip()
            out[m.group(1)] = (value, pending + [i])
            pending = []
        elif not stripped:
            pending = []
    return out


def merge_distances(src_conf, dest_conf):
    """Append entries missing locally (with their comment lines); never rewrite."""
    src_conf, dest_conf = Path(src_conf), Path(dest_conf)
    result = dict(added=[], skipped=[], conflicts=[])
    src_text = src_conf.read_text(encoding="utf-8")
    theirs = _entries(src_text)
    if not dest_conf.exists():
        dest_conf.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_conf, dest_conf)
        result["added"] = sorted(theirs)
        return result
    dest_text = dest_conf.read_text(encoding="utf-8")
    if "[distances]" not in dest_text:
        raise ValueError(f"{dest_conf} has no [distances] section")
    mine = _entries(dest_text)
    src_lines = src_text.splitlines()
    new_lines = []
    for key, (value, lines) in theirs.items():
        if key not in mine:
            new_lines += [src_lines[i] for i in lines]
            result["added"].append(key)
        elif mine[key][0].replace(" ", "") == value.replace(" ", ""):
            result["skipped"].append(key)
        else:
            result["conflicts"].append(
                f"{key}: local '{mine[key][0]}' vs shared '{value}'; local kept"
            )
    if new_lines:
        sep = "" if dest_text.endswith("\n") else "\n"
        dest_conf.write_text(dest_text + sep + "\n".join(new_lines) + "\n", encoding="utf-8")
    return result


def fetch_drive(folder, output, dest, gdown=None, attempts=3, wait=5.0, sleep=None):
    """Download a shared Drive folder file by file.

    Scenes that already exist in ``dest`` are not downloaded again, which keeps
    the number of Drive requests (and rate-limit failures) low. A scene with any
    failed file is removed from ``output`` so it can never be imported half-done.
    Returns dict(failed=[scenes], present=[scenes], duplicates=[paths]).
    """
    import time

    if gdown is None:
        import gdown
    sleep = sleep or time.sleep
    output, dest = Path(output), Path(dest)
    kwargs = dict(url=folder) if folder.startswith("http") else dict(id=folder)
    listing = gdown.download_folder(
        output=str(output), quiet=True, skip_download=True, **kwargs
    )
    if not listing:
        raise RuntimeError("Google Drive returned no files for this folder")

    wanted, duplicates = {}, []
    for item in listing:
        rel = Path(item.local_path).resolve().relative_to(output.resolve())
        if rel in wanted:
            duplicates.append(str(rel))
            continue
        scenes = [p for p in rel.parts[:-1] if SCENE_RE.fullmatch(p)]
        if scenes or rel.name == "distances.conf":
            wanted[rel] = (item.id, scenes[0] if scenes else None)

    result = dict(failed=set(), present=set(), duplicates=sorted(set(duplicates)))
    for rel, (file_id, scene) in sorted(wanted.items()):
        if scene and (dest / scene).exists():
            result["present"].add(scene)
            continue
        if scene in result["failed"]:
            continue
        target = output / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        for attempt in range(1, attempts + 1):
            try:
                if gdown.download(id=file_id, output=str(target), quiet=True) and target.is_file():
                    break
            except Exception:
                pass
            if attempt < attempts:
                sleep(wait * attempt)
        else:
            if scene is None:
                raise RuntimeError(f"Could not download {rel} from Google Drive")
            result["failed"].add(scene)
            shutil.rmtree(target.parent, ignore_errors=True)
    for scene in result["failed"]:
        for d in output.rglob(scene):
            if d.is_dir():
                shutil.rmtree(d, ignore_errors=True)
    result["failed"] = sorted(result["failed"])
    result["present"] = sorted(result["present"])
    return result
