import json
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts/phone"))

from close_range.common import phone_distance
from close_range.phone_checks import check_orientation, scene_images
import annotate


class Temp(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()


class DistanceUnitTests(Temp):
    def conf(self, **values):
        path = self.root / "distances.conf"
        lines = ["[distances]"] + [f"{k} = {v}" for k, v in values.items()]
        path.write_text("\n".join(lines) + "\n")
        return path

    def test_units_convert_to_inch(self):
        path = self.conf(a="17", b="43.18cm", c="431.8 mm", d="0.4318m", e="17in")
        for scene in "abcde":
            self.assertAlmostEqual(phone_distance(scene, path)[0], 17.0)

    def test_inline_comment_allowed(self):
        path = self.conf(a="25cm   # transparent bottle")
        self.assertAlmostEqual(phone_distance("a", path)[0], 25 / 2.54)

    def test_bad_values_rejected(self):
        path = self.conf(a="10ft", b="-3cm", c="0cm", d="abc")
        for scene in "abcd":
            with self.assertRaises(ValueError):
                phone_distance(scene, path)


class PhotoCheckTests(Temp):
    def test_heic_is_not_silently_ignored(self):
        Image.new("RGB", (4, 4)).save(self.root / "view_001.jpg")
        (self.root / "view_002.HEIC").write_bytes(b"")
        with self.assertRaisesRegex(ValueError, "view_002.HEIC"):
            scene_images(self.root)

    def test_hidden_and_other_files_ignored(self):
        Image.new("RGB", (4, 4)).save(self.root / "view_001.jpg")
        (self.root / ".DS_Store").write_bytes(b"")
        (self.root / "notes.txt").write_text("x")
        self.assertEqual([f.name for f in scene_images(self.root)], ["view_001.jpg"])

    def test_mixed_orientation_rejected_unless_allowed(self):
        names, sizes = ["a.jpg", "b.jpg"], [(400, 300), (300, 400)]
        with self.assertRaisesRegex(ValueError, "b.jpg=portrait"):
            check_orientation(names, sizes)
        self.assertEqual(check_orientation(names, sizes, allow_mixed=True), "mixed")
        self.assertEqual(check_orientation(names, [(400, 300)] * 2), "landscape")


class AnnotateDiscoveryTests(Temp):
    def scene(self, name, image=True, annotated=False):
        manifests = self.root / annotate.MANIFESTS
        manifests.mkdir(parents=True, exist_ok=True)
        img = f"data/processed/phone/{name}/view_001.png"
        if image:
            (self.root / img).parent.mkdir(parents=True)
            Image.new("RGB", (4, 4)).save(self.root / img)
        ann = f"data/annotations/phone/{name}.json"
        if annotated:
            (self.root / ann).parent.mkdir(parents=True, exist_ok=True)
            (self.root / ann).write_text("{}")
        (manifests / f"{name}.json").write_text(
            json.dumps(
                dict(scene_id=name, views=[dict(image=img)], reference_index=0, annotation=ann)
            )
        )

    def test_pending_skips_done_and_missing(self):
        self.scene("scene-001", image=False)
        self.scene("scene-002", annotated=True)
        self.scene("scene-003")
        pending, skipped = annotate.pending_manifests(self.root)
        self.assertEqual([p.stem for p in pending], ["scene-003"])
        self.assertEqual(
            dict(skipped),
            {"scene-001": "reference image missing", "scene-002": "already annotated"},
        )

    def test_scene_name_resolves_to_manifest(self):
        self.scene("scene-004")
        self.assertEqual(
            annotate.scene_manifest("scene-004", self.root),
            Path("data/manifests/phone/scene-004.json"),
        )
        for bad in ["", "..", "a/b"]:
            with self.assertRaises(ValueError):
                annotate.scene_manifest(bad, self.root)
        with self.assertRaisesRegex(ValueError, "prepare.py --scene scene-404"):
            annotate.scene_manifest("scene-404", self.root)


if __name__ == "__main__":
    unittest.main()
