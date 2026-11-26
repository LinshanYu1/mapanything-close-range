"""批量评估：同一批场景清单 × 多个模型 × 单图/多图，每个模型只加载一次。

示例：
  bash scripts/local.sh -m close_range.evaluation.run --name phone --manifests 'data/manifests/phone/scene-*.json'
  # 默认用 configs/mixed_split.json 的 test 集
  bash scripts/local.sh -m close_range.evaluation.run --name test --models baseline A B AB
  # 换一套权重时一定换 --name，否则会复用旧结果
  bash scripts/local.sh -m close_range.evaluation.run --name phone_mixed --manifests 'data/manifests/phone/scene-*.json' \\
      --models baseline AB --dir-AB checkpoints/mixed_AB_run1

中断后用同一条命令重跑，已完成的组合会直接复用。
结果：outputs/eval/<name>/results.csv（每个点对一行），预测保存在 predictions/ 下。
"""
import argparse
import sys
import traceback

from close_range.common import ROOT, read
from close_range.evaluation import common
from close_range.evaluation.common import MODELS

FIELDS = ["source", "scene", "model", "mode", "n_views", "pair_id", "gt_cm", "pred_cm",
          "abs_error_cm", "rel_error_pct", "valid", "success_10pct", "status", "reason"]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--name", required=True, help="本次评估的名字，结果放在 outputs/eval/<name>/")
    p.add_argument("--models", nargs="+", default=["baseline", "A", "B"], choices=MODELS)
    p.add_argument("--modes", nargs="+", default=["single", "multi"], choices=["single", "multi"])
    common.add_common_args(p)
    args = p.parse_args(argv)

    paths = common.manifest_list(args)
    out = ROOT / "outputs" / "eval" / args.name
    out.mkdir(parents=True, exist_ok=True)
    results = out / "results.csv"
    key = lambda r: (r["source"], r["scene"], r["model"], r["mode"], str(r["pair_id"]))
    rows, failed = [], 0
    print(f"{len(paths)} 个场景 × {len(args.models)} 个模型 × {len(args.modes)} 种输入", flush=True)
    for variant in args.models:
        model = None
        for i, path in enumerate(paths, 1):
            manifest = read(path)
            for mode in args.modes:
                target = out / "predictions" / variant / mode / manifest["scene_id"]
                tag = f"[{variant} {mode} {i}/{len(paths)}] {manifest['scene_id']}"
                try:
                    if model is None and not (target / "metrics.json").is_file():
                        model = common.load_variant(variant, args)
                    metrics = common.predict_and_score(model, manifest, target, args.device, mode)
                    new = common.pair_rows(manifest, metrics, model=variant, mode=mode, n_views=1 if mode == "single" else len(manifest["views"]))
                    rows += new
                    errs = [r["rel_error_pct"] for r in new if r["rel_error_pct"] != ""]
                    print(f"{tag}: {len(new)} 对点，有效 {len(errs)}" +
                          (f"，平均相对误差 {sum(errs)/len(errs):.1f}%" if errs else ""), flush=True)
                except KeyboardInterrupt:
                    common.write_csv(results, rows, FIELDS, key)
                    raise
                except Exception as error:
                    failed += 1
                    traceback.print_exc()
                    rows.append(common.failed_row(manifest, path, error, model=variant, mode=mode, n_views=""))
                    print(f"{tag}: 失败，继续下一个", flush=True)
                common.write_csv(results, rows, FIELDS, key)
        del model
    print(f"结果：{results}（失败 {failed} 个）")
    print(f"汇总：bash scripts/local.sh -m close_range.evaluation.summarize {results.relative_to(ROOT)}")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
