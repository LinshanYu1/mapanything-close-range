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


_UNIT_TO_INCH = {
    "": 1.0, "in": 1.0, "inch": 1.0, "inches": 1.0,
    "cm": 1 / 2.54, "mm": 1 / 25.4, "m": 100 / 2.54,
}


def phone_distance(scene, config=None):
    """Read a strictly positive measurement for a scene, returned in inch.

    Values may carry a unit suffix: ``43.2cm``, ``432 mm``, ``0.432m``,
    ``17in``. A bare number is treated as inch (backwards compatible).
    """
    import configparser, math, re

    path = resolve(config or "data/raw/phone/distances.conf")
    parser = configparser.ConfigParser(interpolation=None, inline_comment_prefixes=("#",))
    parser.optionxform = str
    with path.open() as f:
        parser.read_file(f)
    if not parser.has_option("distances", scene):
        raise ValueError(f"Missing distance for {scene} in {path}")
    raw = parser.get("distances", scene).strip()
    match = re.fullmatch(r"([0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)\s*([a-zA-Z]*)", raw)
    unit = match.group(2).lower() if match else None
    if not match or unit not in _UNIT_TO_INCH:
        raise ValueError(
            f"Distance for {scene} must be a number with optional unit "
            f"(in, cm, mm, m), got {raw!r}"
        )
    value = float(match.group(1)) * _UNIT_TO_INCH[unit]
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"Distance for {scene} must be positive and finite")
    return value, path
