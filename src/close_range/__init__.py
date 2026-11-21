"""Shared utilities and the pinned local UniCeption dependency."""
from pathlib import Path
import sys

_UNICEPTION_ROOT = Path(__file__).resolve().parents[2] / "third_party" / "uniception"
_loaded = sys.modules.get("uniception")
if _loaded is not None and not Path(_loaded.__file__).resolve().is_relative_to(_UNICEPTION_ROOT):
    raise RuntimeError("External UniCeption is already imported. Restart Python and import close_range first.")
if not (_UNICEPTION_ROOT / "uniception" / "__init__.py").is_file():
    raise RuntimeError("Missing third_party/uniception: use a complete project checkout.")
sys.path.insert(0, str(_UNICEPTION_ROOT))
