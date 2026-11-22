"""批量评估：同一批场景清单 × 多个模型 × 单图/多图，每个模型只加载一次。

示例：
  # 手机照片（当前模型都没用手机照片训练，全部手机场景都可当测试集）
  bash scripts/local.sh scripts/eval/run_eval.py --name phone --manifests 'data/manifests/phone/scene-*.json'
  # GraspNet 测试集（configs/graspnet_split.json 的 test 列表）
  bash scripts/local.sh scripts/eval/run_eval.py --name graspnet_test --split configs/graspnet_split.json --set test

中断后用同一条命令重跑，已完成的组合会直接复用，不重复预测。
结果：outputs/eval/<name>/results.csv（每个点对一行），预测保存在 predictions/ 下。
"""
import argparse
import csv
import glob
import shutil
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from close_range.common import read, resolve  # noqa: E402

FIELDS = ["source", "scene", "model", "mode", "n_views", "pair_id", "gt_cm", "pred_cm",
          "abs_error_cm", "rel_error_pct", "valid", "success_10pct", "status", "reason"]
DEFAULT_DIRS = dict(A="checkpoints/graspnet_A", B="checkpoints/graspnet_B", AB="checkpoints/graspnet_AB")


def manifest_list(args):
    if args.split:
        split = read(resolve(args.split))
        if args.set not in split:
            raise SystemExit(f"{args.split} 里没有 {args.set} 集合")
        paths = [resolve(p) for p in split[args.set]]
    else:
        paths = sorted({Path(p).resolve() for pattern in args.manifests for p in glob.glob(str(resolve(pattern)))})
    if not paths:
        raise SystemExit("没有找到场景清单")
    missing = [p for p in paths if not p.is_file()]
    if missing:
        raise SystemExit(f"{len(missing)} 个清单不存在，例如 {missing[0]}；先下载并预处理这些场景")
    return paths


def load(variant, args):
    if variant == "baseline":
        from run_baseline import load_model
        return load_model(args.device)
    from close_range.models.model_io import load_model
    directory = resolve(getattr(args, f"dir_{variant}"))
    config = read(directory / "model_config.json")
    if config["variant"] != variant:
        raise ValueError(f"{directory} 里的模型类型是 {config['variant']}，不是 {variant}")
    print(f"Loading {variant} from {directory} ({args.weights})", flush=True)
    return load_model(directory, device=args.device, weights_name=args.weights)


def rows_from(manifest, variant, mode, metrics, n_views):
    rows = []
    for r in metrics["results"]:
        rows.append(dict(
            source=manifest["source"], scene=manifest["scene_id"], model=variant, mode=mode,
            n_views=n_views, pair_id=r["pair_id"], gt_cm=round(r["reference_cm"], 3),
            pred_cm="" if r["predicted_cm"] is None else round(r["predicted_cm"], 3),
            abs_error_cm="" if r["absolute_error_cm"] is None else round(r["absolute_error_cm"], 3),
            rel_error_pct="" if r["relative_error_pct"] is None else round(r["relative_error_pct"], 3),
            valid=r["prediction_valid"], success_10pct=r["success_10pct"], status="ok",
            reason=r.get("reason", "")))
    return rows


def write(path, rows):
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(sorted(rows, key=lambda r: (r["source"], r["scene"], r["model"], r["mode"], str(r["pair_id"]))))
    tmp.replace(path)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--name", required=True, help="本次评估的名字，结果放在 outputs/eval/<name>/")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--manifests", nargs="+", help="清单路径或通配符，记得加引号")
    src.add_argument("--split", help="划分文件，例如 configs/graspnet_split.json")
    p.add_argument("--set", default="test", help="配合 --split 使用：train / val / test")
    p.add_argument("--models", nargs="+", default=["baseline", "A", "B"], choices=["baseline", "A", "B", "AB"])
    p.add_argument("--modes", nargs="+", default=["single", "multi"], choices=["single", "multi"])
    p.add_argument("--weights", default="model_best.pt", choices=["model_best.pt", "model.pt"])
    p.add_argument("--device", default="mps", choices=["mps", "cuda", "cpu"])
    for v in ("A", "B", "AB"):
        p.add_argument(f"--dir-{v}", dest=f"dir_{v}", default=DEFAULT_DIRS[v])
    args = p.parse_args()

    from evaluate import evaluate_run
    from run_baseline import predict_scene, prepare_inputs

    paths = manifest_list(args)
    out = ROOT / "outputs" / "eval" / args.name
    out.mkdir(parents=True, exist_ok=True)
    results = out / "results.csv"
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
                    n_views = 1 if mode == "single" else len(manifest["views"])
                    if not (target / "metrics.json").is_file():
                        if target.exists():
                            shutil.rmtree(target)  # 上次中断留下的不完整结果
                        if model is None:
                            model = load(variant, args)
                        prepared = prepare_inputs(manifest, mode)
                        predict_scene(model, manifest, mode, args.device, target, prepared)
                        evaluate_run(manifest, target)
                    new = rows_from(manifest, variant, mode, read(target / "metrics.json"), n_views)
                    rows += new
                    errs = [r["rel_error_pct"] for r in new if r["rel_error_pct"] != ""]
                    print(f"{tag}: {len(new)} 对点，有效 {len(errs)}" +
                          (f"，平均相对误差 {sum(errs)/len(errs):.1f}%" if errs else ""), flush=True)
                except KeyboardInterrupt:
                    write(results, rows)
                    raise
                except Exception as error:
                    failed += 1
                    traceback.print_exc()
                    rows.append(dict(source=manifest.get("source", ""), scene=manifest.get("scene_id", path.stem),
                                     model=variant, mode=mode, n_views="", pair_id="", gt_cm="", pred_cm="",
                                     abs_error_cm="", rel_error_pct="", valid=False, success_10pct=False,
                                     status="failed", reason=f"{type(error).__name__}: {error}"))
                    print(f"{tag}: 失败，继续下一个", flush=True)
                write(results, rows)
        del model
    print(f"结果：{results}（失败 {failed} 个）")
    print(f"汇总：bash scripts/local.sh scripts/eval/summarize.py {results.relative_to(ROOT)}")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
