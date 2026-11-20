import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from types import SimpleNamespace

from close_range.phone_import import (
    fetch_drive,
    find_root,
    import_scenes,
    merge_distances,
    safe_extract,
)


class ImportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.shared = self.root / "Downloads" / "phone_zongyu"
        self.dest = self.root / "raw"

    def tearDown(self):
        self.temp.cleanup()

    def scene(self, base, name, files=("view_001.jpg", "view_002.jpg"), data=b"x"):
        d = base / name
        d.mkdir(parents=True)
        for f in files:
            (d / f).write_bytes(data)
        return d

    def test_find_root_through_wrapper_folders(self):
        self.scene(self.shared, "scene-002")
        self.assertEqual(find_root(self.root), self.shared)
        with self.assertRaises(ValueError):
            find_root(self.dest_empty())

    def dest_empty(self):
        d = self.root / "empty"
        d.mkdir()
        return d

    def test_import_adds_skips_and_reports_conflicts(self):
        self.scene(self.shared, "scene-002")
        self.scene(self.shared, "scene-003")
        self.scene(self.shared, "scene-004", files=("view_001.jpg", "view_002.jpg", "view_003.jpg"))
        (self.shared / "scene-002" / ".DS_Store").write_bytes(b"junk")
        self.scene(self.dest, "scene-003")  # identical -> skipped
        self.scene(self.dest, "scene-004")  # different -> conflict
        result = import_scenes(self.shared, self.dest)
        self.assertEqual(result["added"], ["scene-002"])
        self.assertEqual(result["skipped"], ["scene-003"])
        self.assertEqual(len(result["conflicts"]), 1)
        self.assertEqual(
            sorted(p.name for p in (self.dest / "scene-002").iterdir()),
            ["view_001.jpg", "view_002.jpg"],
        )
        self.assertEqual(len(list((self.dest / "scene-004").iterdir())), 2)
        self.assertEqual([p.name for p in self.dest.iterdir() if p.name.startswith(".")], [])

    def test_merge_appends_new_entries_with_comments(self):
        shared = self.root / "shared.conf"
        shared.write_text(
            "[distances]\n# scene-002 note\nscene-002 = 9.843\n\n"
            "# scene-011 transparent bag\nscene-011 = 28cm\nscene-012 = 68 cm  # inline\n"
            "scene-013 = 50cm\n"
        )
        local = self.root / "local.conf"
        local.write_text("[distances]\nscene-002 = 9.843\nscene-013 = 48cm")
        result = merge_distances(shared, local)
        self.assertEqual(result["added"], ["scene-011", "scene-012"])
        self.assertEqual(result["skipped"], ["scene-002"])
        self.assertIn("scene-013", result["conflicts"][0])
        text = local.read_text()
        self.assertIn("# scene-011 transparent bag\nscene-011 = 28cm\n", text)
        self.assertIn("scene-013 = 48cm\n", text)
        self.assertEqual(text.count("[distances]"), 1)

    def test_merge_copies_when_local_missing(self):
        shared = self.root / "shared.conf"
        shared.write_text("[distances]\nscene-002 = 25cm\n")
        local = self.root / "sub" / "distances.conf"
        self.assertEqual(merge_distances(shared, local)["added"], ["scene-002"])
        self.assertEqual(local.read_text(), shared.read_text())

    def test_zip_extract_rejects_escape_and_skips_macosx(self):
        good = self.root / "good.zip"
        with zipfile.ZipFile(good, "w") as zf:
            zf.writestr("phone_zongyu/scene-002/view_001.jpg", b"x")
            zf.writestr("__MACOSX/phone_zongyu/._view_001.jpg", b"x")
        out = safe_extract(good, self.root / "out")
        self.assertEqual(find_root(out), out / "phone_zongyu")
        self.assertFalse((out / "__MACOSX").exists())
        bad = self.root / "bad.zip"
        with zipfile.ZipFile(bad, "w") as zf:
            zf.writestr("../evil.txt", b"x")
        with self.assertRaises(ValueError):
            safe_extract(bad, self.root / "out2")
        self.assertFalse((self.root / "evil.txt").exists())


class FakeDrive:
    """Stands in for gdown: a listing plus per-id download behaviour."""

    def __init__(self, files, fail=()):
        self.files, self.fail, self.calls = files, set(fail), []

    def download_folder(self, output, quiet, skip_download, **kwargs):
        assert skip_download
        return [
            SimpleNamespace(id=fid, path=rel, local_path=str(Path(output) / rel))
            for fid, rel in self.files
        ]

    def download(self, id, output, quiet):
        self.calls.append(id)
        if id in self.fail:
            raise RuntimeError("rate limited")
        Path(output).write_bytes(id.encode())
        return output


class FetchDriveTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.out, self.dest = self.root / "drive", self.root / "raw"
        (self.dest / "scene-002").mkdir(parents=True)

    def tearDown(self):
        self.temp.cleanup()

    def test_skips_local_retries_and_drops_incomplete_scenes(self):
        drive = FakeDrive(
            [
                ("a1", "scene-002/view_001.jpg"),
                ("b1", "scene-011/view_001.jpg"),
                ("b1dup", "scene-011/view_001.jpg"),
                ("c1", "scene-012/view_001.jpg"),
                ("c2", "scene-012/view_002.jpg"),
                ("d", "distances.conf"),
                ("e", "summary.csv"),
            ],
            fail={"c2"},
        )
        waits = []
        result = fetch_drive("ID", self.out, self.dest, gdown=drive, sleep=waits.append)
        self.assertEqual(result["present"], ["scene-002"])
        self.assertEqual(result["failed"], ["scene-012"])
        self.assertEqual(result["duplicates"], ["scene-011/view_001.jpg"])
        self.assertNotIn("a1", drive.calls)
        self.assertNotIn("e", drive.calls)
        self.assertEqual(drive.calls.count("c2"), 3)
        self.assertEqual(len(waits), 2)
        self.assertTrue((self.out / "scene-011" / "view_001.jpg").is_file())
        self.assertFalse((self.out / "scene-012").exists())
        self.assertEqual(import_scenes(find_root(self.out), self.dest)["added"], ["scene-011"])

    def test_missing_distances_conf_is_fatal(self):
        drive = FakeDrive([("d", "distances.conf")], fail={"d"})
        with self.assertRaises(RuntimeError):
            fetch_drive("ID", self.out, self.dest, gdown=drive, sleep=lambda s: None)


if __name__ == "__main__":
    unittest.main()
