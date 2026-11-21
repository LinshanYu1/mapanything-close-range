"""Input checks for phone scenes that need no model code."""

from pathlib import Path

SUPPORTED_SUFFIXES = {".jpg", ".jpeg", ".png"}
# Photo formats that would otherwise be silently ignored by the scene scan.
UNSUPPORTED_PHOTO_SUFFIXES = {".heic", ".heif", ".dng", ".webp", ".tif", ".tiff"}


def scene_images(raw):
    """Return sorted supported images; refuse scenes with skipped photo files."""
    raw = Path(raw)
    files = sorted(
        f
        for f in raw.iterdir()
        if f.is_file() and not f.name.startswith(".")
    )
    unsupported = [f.name for f in files if f.suffix.lower() in UNSUPPORTED_PHOTO_SUFFIXES]
    if unsupported:
        raise ValueError(
            "Unsupported photo files would be ignored: "
            f"{', '.join(unsupported)}. Convert them to JPG/PNG or remove them."
        )
    images = [f for f in files if f.suffix.lower() in SUPPORTED_SUFFIXES]
    if not images:
        raise ValueError("No JPG/PNG input images found")
    return images


def orientation(size):
    width, height = size
    if width > height:
        return "landscape"
    if height > width:
        return "portrait"
    return "square"


def check_orientation(names, sizes, allow_mixed=False):
    """Mixed landscape/portrait scenes share one crop and can lose endpoints."""
    kinds = {name: orientation(size) for name, size in zip(names, sizes)}
    found = {k for k in kinds.values() if k != "square"}
    if len(found) > 1 and not allow_mixed:
        detail = ", ".join(f"{name}={kind}" for name, kind in kinds.items())
        raise ValueError(
            "Mixed landscape/portrait images in one scene (after EXIF rotation): "
            f"{detail}. All images are cropped to one shared size, which can cut off "
            "the marked endpoints. Remove or reshoot the odd ones, or pass "
            "--allow-mixed-orientation deliberately."
        )
    return found.pop() if len(found) == 1 else ("mixed" if found else "square")
