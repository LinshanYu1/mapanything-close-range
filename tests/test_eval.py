import csv
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from close_range.evaluation import run, stability, summarize  # noqa: E402


def fakes(calls, scale):
    def predict(model, manifest, mode, device, target, prepared):
        calls.append((manifest["scene_id"], mode, tuple(v["id"] for v in manifest["views"])))
        target.mkdir(parents=True)

    def evaluate(manifest, target):
        gt = manifest["gt"]
        n = len(manifest["views"])
        pred = None if manifest["scene_id"] == "bad" else gt * scale(n)
        err = None if pred is None else abs(pred - gt)
        r = dict(pair_id="pair_001", reference_cm=gt, predicted_cm=pred, absolute_error_cm=err,
                 relative_error_pct=None if pred is None else err / gt * 100,
                 prediction_valid=pred is not None, success_10pct=pred is not None and err / gt <= 0.1, reason="")
        (target / "metrics.json").write_text(json.dumps(dict(results=[r])))

    return {"run_baseline": types.SimpleNamespace(predict_scene=predict, prepare_inputs=lambda m, mode: None,
                                                  load_model=lambda device: "model"),
            "evaluate": types.SimpleNamespace(evaluate_run=evaluate)}


def manifests(tmp, views=3):
    for name, gt in [("s1", 8.0), ("s2", 70.0), ("bad", 30.0)]:
        (tmp / f"{name}.json").write_text(json.dumps(dict(
            scene_id=name, source="phone", gt=gt, reference_index=0,
            views=[dict(id=f"view_{i:03d}") for i in range(1, views + 1)])))


class EvalTest(unittest.TestCase):
    def test_run_resume_and_summary(self):
        calls = []
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            manifests(tmp, views=2)
            argv = ["--name", "t", "--manifests", str(tmp / "*.json"), "--models", "baseline"]
            with mock.patch.dict(sys.modules, fakes(calls, lambda n: 1.05)), mock.patch.object(run, "ROOT", tmp):
                run.main(argv)
                self.assertEqual(len(calls), 6)
                run.main(argv)  # 第二次全部复用
                self.assertEqual(len(calls), 6)
            rows = list(csv.DictReader((tmp / "outputs/eval/t/results.csv").open()))
            self.assertEqual(len(rows), 6)
            table = summarize.table(rows)
            overall = [r for r in table if r["bin"] == "all" and r["mode"] == "multi"][0]
            self.assertEqual((overall["pairs"], overall["invalid"]), (3, 1))
            self.assertAlmostEqual(overall["within_10pct"], 200 / 3)
            bins = {r["bin"]: r["pairs"] for r in table if r["mode"] == "single"}
            self.assertEqual(bins, {"all": 3, "<20cm": 1, "20-60cm": 1, ">=60cm": 1})

    def test_default_split(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            manifests(tmp)
            (tmp / "cfg").mkdir()
            split = tmp / "cfg" / "split.json"
            split.write_text(json.dumps(dict(test=[str(tmp / "s1.json")], train=[str(tmp / "s2.json")])))
            import argparse
            from close_range.evaluation import common
            p = argparse.ArgumentParser()
            common.add_common_args(p)
            self.assertEqual(p.parse_args([]).split, "configs/mixed_split.json")
            with mock.patch.object(common, "resolve", lambda x: Path(x)):
                self.assertEqual(common.manifest_list(p.parse_args(["--split", str(split)])), [tmp / "s1.json"])
                # 手机全集里去掉训练用过的 s2
                args = p.parse_args(["--manifests", str(tmp / "*.json"), "--exclude-seen", str(split)])
                with mock.patch.object(common, "resolve", lambda x: Path(x)):
                    got = common.manifest_list(args)
                self.assertEqual(sorted(x.name for x in got), ["bad.json", "s1.json"])

    def test_subsets(self):
        m = dict(scene_id="s", reference_index=1, views=[{}] * 5)
        all_subs = stability.subsets(m, [2, 3], 0, 0)
        self.assertEqual(len(all_subs), 4 + 6 + 1)  # 加上全部视图
        self.assertEqual(all_subs[-1], [1, 0, 2, 3, 4])
        self.assertTrue(all(s[0] == 1 and 1 not in s[1:] for s in all_subs))
        capped = stability.subsets(m, [2, 3], 4, 0)
        self.assertEqual(capped, stability.subsets(m, [2, 3], 4, 0))  # 固定种子可复现
        self.assertEqual(len(capped), 5)
        self.assertEqual(capped[-1], [1, 0, 2, 3, 4])  # 抽样不会丢掉全部视图
        self.assertEqual(len(stability.subsets(m, [2, 3], 0, 0, full=False)), 10)
        # 三张图（GraspNet）：两个双图子集 + 全部视图，不重复
        self.assertEqual(stability.subsets(dict(m, views=[{}] * 3), [2, 3], 0, 0), [[1, 0], [1, 2], [1, 0, 2]])
        self.assertEqual(stability.subsets(dict(m, views=[{}] * 2), [2, 3], 0, 0), [[1, 0]])

    def test_stability_run_and_metrics(self):
        calls = []
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            manifests(tmp, views=4)  # 参考图 + 3 张辅助图：3 个双图子集 + 3 个三图子集
            argv = ["--name", "t", "--manifests", str(tmp / "*.json"), "--models", "baseline", "--no-full"]
            # 双图预测偏大 10%，三图偏大 30%
            with mock.patch.dict(sys.modules, fakes(calls, lambda n: 1.1 if n == 2 else 1.3)), \
                 mock.patch.object(stability, "ROOT", tmp):
                stability.main(argv)
                self.assertEqual(len(calls), 3 * 6)
                self.assertTrue(all(c[2][0] == "view_001" for c in calls))
                stability.main(argv)
                self.assertEqual(len(calls), 3 * 6)
            with (tmp / "outputs/eval/t/stability/results.csv").open() as f:
                rows = list(csv.DictReader(f))
            stats = {s["scene"]: s for s in stability.scene_stats(rows)}
            self.assertNotIn("cv_pct", stats["bad"])
            s1 = stats["s1"]
            self.assertAlmostEqual(s1["cv_pct"], 0.1 / 1.2 * 100)
            self.assertAlmostEqual(s1["range_gt_pct"], 20)
            self.assertAlmostEqual(s1["mean_rel_err_pct"], 20)
            top = [r for r in stability.summarize(stability.scene_stats(rows)) if r["bin"] == "all"][0]
            self.assertEqual((top["pairs"], top["excluded_pairs"]), (2, 1))
            self.assertTrue((tmp / "outputs/eval/t/stability/stability_summary.csv").is_file())


if __name__ == "__main__":
    unittest.main()
