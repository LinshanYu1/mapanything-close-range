import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import predict_directory as batch


class BatchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.inputs = self.root / "phone"
        self.inputs.mkdir()
        self.manifests = self.root / "manifests"
        self.manifests.mkdir()

    def tearDown(self):
        self.temp.cleanup()

    def scene(self, name, count):
        scene = self.inputs / name
        scene.mkdir()
        for n in range(count):
            Image.new("RGB", (28, 28)).save(scene / f"view_{n}.png")
        return scene

    def test_modes_one_model_and_failure_isolation(self):
        self.scene("a-single", 1)
        self.scene("b-empty", 0)
        self.scene("c-multi", 3)
        with patch.object(batch, "prepare_inputs", return_value=()), patch.object(
            batch, "load_model", return_value=object()
        ) as load, patch.object(batch, "predict_scene") as predict:
            records = batch.run_batch(
                self.inputs, self.manifests, self.root / "out", "cpu"
            )
        self.assertEqual([r["mode"] for r in records], ["single", "", "multi"])
        self.assertEqual(records[1]["status"], "failed")
        self.assertEqual(records[2]["status"], "predicted_without_labels")
        self.assertEqual(load.call_count, 1)
        self.assertEqual(predict.call_count, 2)
        self.assertTrue((self.root / "out/summary.csv").is_file())

    def test_manifest_cannot_silently_omit_images(self):
        scene = self.scene("one", 2)
        batch.save(self.manifests / "one.json", dict(scene_id="one", views=[]))
        with self.assertRaisesRegex(ValueError, "differ from manifest"):
            batch.scene_manifest(scene, self.manifests)

    def test_check_only_does_not_load_model(self):
        self.scene("one", 1)
        with patch.object(batch, "prepare_inputs", return_value=()), patch.object(
            batch, "load_model"
        ) as load:
            records = batch.run_batch(
                self.inputs, self.manifests, self.root / "out", "cpu", True
            )
        load.assert_not_called()
        self.assertEqual(records[0]["status"], "checked")

    def test_no_output_overwrite(self):
        self.scene("one", 1)
        out = self.root / "out"
        out.mkdir()
        with self.assertRaises(FileExistsError):
            batch.run_batch(self.inputs, self.manifests, out, "cpu", True)

    def test_invalid_raw_dimensions_rejected(self):
        scene = self.scene("one", 1)
        Image.new("RGB", (1702, 1276)).save(scene / "view_0.png")
        with self.assertRaisesRegex(ValueError, "preprocessed"):
            batch.scene_manifest(scene, self.manifests)


if __name__ == "__main__":
    unittest.main()
