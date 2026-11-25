import csv
import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts/eval" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class EvalTest(unittest.TestCase):
    def test_run_resume_and_summary(self):
        run_eval, summarize = load("run_eval"), load("summarize")
        calls = []

        def predict(model, manifest, mode, device, target, prepared):
            calls.append((manifest["scene_id"], mode))
            target.mkdir(parents=True)

        def evaluate(manifest, target):
            gt = manifest["gt"]
            pred = None if manifest["scene_id"] == "bad" else gt * 1.05
            r = dict(pair_id="pair_001", reference_cm=gt, predicted_cm=pred,
                     absolute_error_cm=None if pred is None else pred - gt,
                     relative_error_pct=None if pred is None else 5.0,
                     prediction_valid=pred is not None, success_10pct=pred is not None, reason="")
            (target / "metrics.json").write_text(json.dumps(dict(results=[r])))

        fake_rb = types.SimpleNamespace(predict_scene=predict, prepare_inputs=lambda m, mode: None,
                                        load_model=lambda device: "model")
        fake_ev = types.SimpleNamespace(evaluate_run=evaluate)
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            for name, gt in [("s1", 8.0), ("s2", 70.0), ("bad", 30.0)]:
                (tmp / f"{name}.json").write_text(json.dumps(
                    dict(scene_id=name, source="phone", gt=gt, views=[1, 2])))
            argv = ["run_eval", "--name", "t", "--manifests", str(tmp / "*.json"), "--models", "baseline"]
            with mock.patch.dict(sys.modules, run_baseline=fake_rb, evaluate=fake_ev), \
                 mock.patch.object(run_eval, "ROOT", tmp), mock.patch.object(sys, "argv", argv):
                run_eval.main()
                self.assertEqual(len(calls), 6)
                run_eval.main()  # 第二次全部复用
                self.assertEqual(len(calls), 6)
            results = tmp / "outputs/eval/t/results.csv"
            rows = list(csv.DictReader(results.open()))
            self.assertEqual(len(rows), 6)
            table = summarize.table(rows)
            overall = [r for r in table if r["bin"] == "all" and r["mode"] == "multi"][0]
            self.assertEqual((overall["pairs"], overall["invalid"]), (3, 1))
            self.assertAlmostEqual(overall["within_10pct"], 200 / 3)
            bins = {r["bin"]: r["pairs"] for r in table if r["mode"] == "single"}
            self.assertEqual(bins, {"all": 3, "<20cm": 1, "20-60cm": 1, ">=60cm": 1})


if __name__ == "__main__":
    unittest.main()
