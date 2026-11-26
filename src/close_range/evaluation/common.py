"""评估模块共用的函数：找清单、加载模型、预测并评分、写 CSV。"""
import csv
import glob
import shutil
import sys
from pathlib import Path

from close_range.common import ROOT, read, resolve

# 复用 scripts/ 下已验证的推理和评分流程，避免两套规则。
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

DEFAULT_DIRS = dict(A="checkpoints/graspnet_A", B="checkpoints/graspnet_B", AB="checkpoints/graspnet_AB")
MODELS = ["baseline", "A", "B", "AB"]
DEFAULT_SPLIT = "configs/mixed_split.json"


def add_common_args(p):
    src = p.add_mutually_exclusive_group()
    src.add_argument("--manifests", nargs="+", help="清单路径或通配符，记得加引号（例如手机照片，不在任何划分里）")
    src.add_argument("--split", default=DEFAULT_SPLIT, help=f"划分文件（默认 {DEFAULT_SPLIT}）")
    p.add_argument("--set", default="test", help="配合 --split 使用：train / val / test")
    p.add_argument("--exclude-seen", metavar="SPLIT", help="配合 --manifests：去掉这个划分文件 train/val 里出现过的场景")
    p.add_argument("--weights", default="model_best.pt", choices=["model_best.pt", "model.pt"])
    p.add_argument("--device", default="mps", choices=["mps", "cuda", "cpu"])
    for v in ("A", "B", "AB"):
        p.add_argument(f"--dir-{v}", dest=f"dir_{v}", default=DEFAULT_DIRS[v],
                       help=f"{v} 权重目录（默认 {DEFAULT_DIRS[v]}）")


def manifest_list(args):
    if not args.manifests:
        split = read(resolve(args.split))
        if args.set not in split:
            raise SystemExit(f"{args.split} 里没有 {args.set} 集合")
        paths = [resolve(p) for p in split[args.set]]
    else:
        paths = sorted({Path(p).resolve() for pattern in args.manifests
                        for p in glob.glob(str(resolve(pattern)))})
    if getattr(args, "exclude_seen", None) and args.manifests:
        split = read(resolve(args.exclude_seen))
        seen_paths = {resolve(p).resolve() for k in ("train", "val") for p in split.get(k, [])}
        seen_ids = set()
        for p in seen_paths:
            if p.is_file():
                m = read(p)
                seen_ids.add((m.get("source"), m.get("scene_id")))
        kept = []
        for p in paths:
            m = read(p)
            if p.resolve() in seen_paths or (m.get("source"), m.get("scene_id")) in seen_ids:
                continue
            kept.append(p)
        print(f"--exclude-seen：去掉训练/验证用过的 {len(paths) - len(kept)} 个场景，剩 {len(kept)} 个", flush=True)
        paths = kept
    if not paths:
        raise SystemExit("没有找到场景清单")
    missing = [p for p in paths if not p.is_file()]
    if missing:
        raise SystemExit(f"{len(missing)} 个清单不存在，例如 {missing[0]}；先下载并预处理这些场景")
    return paths


def load_variant(variant, args):
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


def predict_and_score(model, manifest, target, device, mode="multi"):
    """single 只用参考图；multi 用 manifest 的全部视图。结果写进 target，已完成则复用。"""
    from evaluate import evaluate_run
    from run_baseline import predict_scene, prepare_inputs
    if not (target / "metrics.json").is_file():
        if target.exists():
            shutil.rmtree(target)  # 上次中断留下的不完整结果
        prepared = prepare_inputs(manifest, mode)
        predict_scene(model, manifest, mode, device, target, prepared)
        evaluate_run(manifest, target)
    return read(target / "metrics.json")


def pair_rows(manifest, metrics, **extra):
    rows = []
    for r in metrics["results"]:
        rows.append(dict(
            source=manifest["source"], scene=manifest["scene_id"], **extra,
            pair_id=r["pair_id"], gt_cm=round(r["reference_cm"], 3),
            pred_cm="" if r["predicted_cm"] is None else round(r["predicted_cm"], 3),
            abs_error_cm="" if r["absolute_error_cm"] is None else round(r["absolute_error_cm"], 3),
            rel_error_pct="" if r["relative_error_pct"] is None else round(r["relative_error_pct"], 3),
            valid=r["prediction_valid"], success_10pct=r["success_10pct"], status="ok",
            reason=r.get("reason", "")))
    return rows


def failed_row(manifest, path, error, **extra):
    return dict(source=manifest.get("source", ""), scene=manifest.get("scene_id", path.stem), **extra,
                pair_id="", gt_cm="", pred_cm="", abs_error_cm="", rel_error_pct="", valid=False,
                success_10pct=False, status="failed", reason=f"{type(error).__name__}: {error}")


def write_csv(path, rows, fields, key):
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(sorted(rows, key=key))
    tmp.replace(path)
