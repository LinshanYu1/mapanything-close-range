from pathlib import Path
import hashlib
import json
import os
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "third_party/map-anything"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "outputs/.matplotlib"))
REVISION = "a1d87e9086706fb9974f3be5a3e3a0ca5401c5aa"


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def resolve(path):
    return ROOT / path


def phone_distance(scene, config=None):
    """Read a strictly positive inch measurement for a scene."""
    import configparser, math

    path = resolve(config or "data/raw/phone/distances.conf")
    parser = configparser.ConfigParser(interpolation=None)
    parser.optionxform = str
    with path.open() as f:
        parser.read_file(f)
    if not parser.has_option("distances", scene):
        raise ValueError(f"Missing distance for {scene} in {path}")
    value = parser.getfloat("distances", scene)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"Distance for {scene} must be positive and finite (inch)")
    return value, path
