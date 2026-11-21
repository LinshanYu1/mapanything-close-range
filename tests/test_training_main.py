"""训练入口的划分隔离与参数检查；不加载大模型。"""
import json
import tempfile
import unittest
from pathlib import Path

from close_range.training.main import load_split, parse_args


class TrainingMainTests(unittest.TestCase):
    def test_variant_defaults(self):
        for variant, weight in (("A", 0), ("B", .1), ("AB", .1)):
            args = parse_args(["--variant", variant, "--split", "x.json", "--output-dir", "x"])
            self.assertEqual(args.consistency_weight, weight)

    def test_split_rejects_same_scene_across_cameras(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for camera in ("realsense", "kinect"):
                (root / f"{camera}.json").write_text(json.dumps({
                    "source": "graspnet", "scene_id": f"scene_0000_{camera}",
                    "original_scene": "scene_0000", "views": [{"sha256": camera}],
                }))
            split = root / "split.json"
            split.write_text(json.dumps({"train": [str(root / "realsense.json")],
                                         "val": [str(root / "kinect.json")]}))
            with self.assertRaisesRegex(ValueError, "原始场景"):
                load_split(split)

    def test_valid_split_and_test_image_leak(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            entries = {}
            for name in ("train", "val", "test"):
                path = root / f"{name}.json"
                path.write_text(json.dumps({"source": "phone", "scene_id": name,
                                            "views": [{"sha256": name}]}))
                entries[name] = [str(path)]
            split = root / "split.json"
            split.write_text(json.dumps(entries))
            self.assertEqual(len(load_split(split)["test"]), 1)
            path.write_text(json.dumps({"source": "phone", "scene_id": "test",
                                       "views": [{"sha256": "train"}]}))
            with self.assertRaisesRegex(ValueError, "图片跨集合"):
                load_split(split)


if __name__ == "__main__":
    unittest.main()
