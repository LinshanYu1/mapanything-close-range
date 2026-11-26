"""跨视角子集稳定性评估：同一场景、同一对端点，换不同的辅助图片组合，预测距离波动多大。

做法与训练时一致：参考图固定不变（标注点在参考图上），从其余视图里取 1 张、2 张……组成不同子集
（默认参考图 + 1 张、参考图 + 2 张，再加上全部视图），每个子集单独预测一次。
子集只由场景和随机种子决定，所以原模型、A、B、AB 用的是完全相同的组合。对每个场景、每个模型、每对点：
  - cv_pct        预测距离的变异系数 = 标准差 / 平均值 × 100（越小越稳定）
  - range_gt_pct  (最大 − 最小) / 真实距离 × 100（波动幅度占真实距离的比例）
  - mean_rel_err  各子集相对误差的平均值
只有至少 2 个子集有效的点对才参与统计。

示例：
  # 默认用 configs/mixed_split.json 的 test 集
  bash scripts/local.sh -m close_range.evaluation.stability --name test_stability --models baseline A B AB
  # 手机照片（不在任何划分里，作为额外的真实场景测试）
  bash scripts/local.sh -m close_range.evaluation.stability --name phone --manifests 'data/manifests/phone/scene-*.json'
  bash scripts/local.sh -m close_range.evaluation.stability --name phone --summarize-only

子集多时很耗时：--sizes 控制子集大小（含参考图），--max-subsets 限制每个场景最多取几个（固定随机种子抽样）。
中断后用同一条命令重跑，已完成的子集直接复用。结果在 outputs/eval/<name>/stability/。
"""
import argparse
import csv
import itertools
import random
import statistics
import sys
import traceback
from collections import defaultdict

from close_range.common import ROOT, read
from close_range.evaluation import common
from close_range.evaluation.common import MODELS
from close_range.evaluation.summarize import BINS

FIELDS = ["source", "scene", "model", "subset", "n_views", "pair_id", "gt_cm", "pred_cm",
          "abs_error_cm", "rel_error_pct", "valid", "success_10pct", "status", "reason"]
SUMMARY = ["source", "model", "bin", "pairs", "median_cv_pct", "mean_cv_pct", "median_range_gt_pct",
           "cv_le_5pct", "median_mean_rel_err_pct", "excluded_pairs"]


def subsets(manifest, sizes, max_subsets, seed, full=True):
    """参考图 + 若干辅助图的组合，返回视图下标列表（参考图在第一个）。
    full=True 时总是加入全部视图；max_subsets 限制的是其余子集的个数。"""
    ref = manifest["reference_index"]
    aux = [i for i in range(len(manifest["views"])) if i != ref]
    combos = [list(c) for s in sorted(sizes) if 1 <= s - 1 <= len(aux)
              for c in itertools.combinations(aux, s - 1)]
    if full and aux:
        combos = [c for c in combos if len(c) < len(aux)]
    if max_subsets and len(combos) > max_subsets:
        rng = random.Random(f"{seed}-{manifest['scene_id']}")
        combos = sorted(rng.sample(combos, max_subsets), key=lambda c: (len(c), c))
    if full and aux:
        combos.append(aux)
    return [[ref] + c for c in combos]


def sub_manifest(manifest, indices):
    m = dict(manifest)
    m["views"] = [manifest["views"][i] for i in indices]
    m["reference_index"] = 0
    return m


def scene_stats(rows):
    """每个 (source, scene, model, pair) 一行稳定性指标。"""
    groups = defaultdict(list)
    for r in rows:
        if r["status"] == "ok":
            groups[(r["source"], r["scene"], r["model"], r["pair_id"])].append(r)
    out = []
    for (source, scene, model, pair), g in sorted(groups.items()):
        ok = [r for r in g if str(r["valid"]) == "True"]
        preds = [float(r["pred_cm"]) for r in ok]
        gt = float(g[0]["gt_cm"])
        item = dict(source=source, scene=scene, model=model, pair_id=pair, gt_cm=gt,
                    subsets=len(g), valid_subsets=len(ok))
        if len(preds) >= 2:
            mean = statistics.mean(preds)
            item.update(mean_pred_cm=mean, cv_pct=statistics.pstdev(preds) / mean * 100,
                        range_gt_pct=(max(preds) - min(preds)) / gt * 100,
                        mean_rel_err_pct=statistics.mean(float(r["rel_error_pct"]) for r in ok))
        out.append(item)
    return out


def summarize(stats):
    groups = defaultdict(list)
    for s in stats:
        groups[(s["source"], s["model"], "all")].append(s)
        for name, lo, hi in BINS:
            if lo <= s["gt_cm"] < hi:
                groups[(s["source"], s["model"], name)].append(s)
    order = {"all": 0, **{b[0]: i + 1 for i, b in enumerate(BINS)}}
    out = []
    for key in sorted(groups, key=lambda k: (k[0], order[k[2]], MODELS.index(k[1]) if k[1] in MODELS else 9)):
        g = groups[key]
        ok = [s for s in g if "cv_pct" in s]
        cvs = [s["cv_pct"] for s in ok]
        med = lambda xs: statistics.median(xs) if xs else None
        out.append(dict(zip(["source", "model", "bin"], key), pairs=len(ok),
                        median_cv_pct=med(cvs), mean_cv_pct=statistics.mean(cvs) if cvs else None,
                        median_range_gt_pct=med([s["range_gt_pct"] for s in ok]),
                        cv_le_5pct=100 * sum(c <= 5 for c in cvs) / len(cvs) if cvs else None,
                        median_mean_rel_err_pct=med([s["mean_rel_err_pct"] for s in ok]),
                        excluded_pairs=len(g) - len(ok)))
    return out


def report(results):
    rows = list(csv.DictReader(results.open()))
    stats = scene_stats(rows)
    table = summarize(stats)
    fmt = lambda v: "-" if v is None else (f"{v:.1f}" if isinstance(v, float) else str(v))
    with (results.parent / "stability_scenes.csv").open("w", newline="") as f:
        cols = ["source", "scene", "model", "pair_id", "gt_cm", "subsets", "valid_subsets",
                "mean_pred_cm", "cv_pct", "range_gt_pct", "mean_rel_err_pct"]
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows({k: fmt(s.get(k)) if isinstance(s.get(k), float) else s.get(k, "") for k in cols} for s in stats)
    with (results.parent / "stability_summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY)
        w.writeheader()
        w.writerows({k: fmt(v) if isinstance(v, float) else v for k, v in r.items()} for r in table)
    head = (f"{'source':9}{'bin':9}{'model':9}{'pairs':>6}{'medCV%':>8}{'meanCV%':>9}"
            f"{'range/gt%':>10}{'CV<=5%':>8}{'err%':>7}{'excl':>6}")
    print(head)
    print("-" * len(head))
    last = None
    for r in table:
        if last and (r["source"], r["bin"]) != last:
            print()
        last = (r["source"], r["bin"])
        print(f"{r['source']:9}{r['bin']:9}{r['model']:9}{r['pairs']:>6}{fmt(r['median_cv_pct']):>8}"
              f"{fmt(r['mean_cv_pct']):>9}{fmt(r['median_range_gt_pct']):>10}{fmt(r['cv_le_5pct']):>8}"
              f"{fmt(r['median_mean_rel_err_pct']):>7}{r['excluded_pairs']:>6}")
    print(f"\n已保存 {results.parent / 'stability_summary.csv'} 和 stability_scenes.csv")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--name", required=True, help="结果放在 outputs/eval/<name>/stability/")
    p.add_argument("--summarize-only", action="store_true", help="不跑模型，只重新汇总已有结果")
    p.add_argument("--models", nargs="+", default=["baseline", "A", "B"], choices=MODELS)
    p.add_argument("--sizes", nargs="+", type=int, default=[2, 3], help="子集大小（含参考图），默认 2 3")
    p.add_argument("--max-subsets", type=int, default=6, help="每个场景最多几个子集，0 表示全部")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-full", dest="full", action="store_false", help="不加入“全部视图”这个子集")
    if "--summarize-only" in (argv if argv is not None else sys.argv[1:]):
        args = p.parse_args(argv)
        return report(ROOT / "outputs" / "eval" / args.name / "stability" / "results.csv")
    common.add_common_args(p)
    args = p.parse_args(argv)
    if any(s < 2 for s in args.sizes):
        raise SystemExit("--sizes 至少为 2（参考图 + 1 张辅助图）")

    paths = common.manifest_list(args)
    out = ROOT / "outputs" / "eval" / args.name / "stability"
    out.mkdir(parents=True, exist_ok=True)
    results = out / "results.csv"
    key = lambda r: (r["source"], r["scene"], r["model"], r["subset"], str(r["pair_id"]))
    plan = [(path, read(path)) for path in paths]
    plan = [(path, m, subsets(m, args.sizes, args.max_subsets, args.seed, args.full)) for path, m in plan]
    skipped = [m["scene_id"] for _, m, s in plan if len(s) < 2]
    plan = [x for x in plan if len(x[2]) >= 2]
    total = sum(len(s) for _, _, s in plan)
    print(f"{len(plan)} 个场景，共 {total} 个子集 × {len(args.models)} 个模型"
          + (f"；视图不够、跳过 {len(skipped)} 个场景" if skipped else ""), flush=True)
    rows, failed = [], 0
    for variant in args.models:
        model, done = None, 0
        for path, manifest, subs in plan:
            for indices in subs:
                done += 1
                m = sub_manifest(manifest, indices)
                tag_name = "+".join(v["id"] for v in m["views"])
                target = out / "predictions" / variant / manifest["scene_id"] / tag_name
                tag = f"[{variant} {done}/{total}] {manifest['scene_id']} {tag_name}"
                try:
                    if model is None and not (target / "metrics.json").is_file():
                        model = common.load_variant(variant, args)
                    metrics = common.predict_and_score(model, m, target, args.device)
                    new = common.pair_rows(manifest, metrics, model=variant, subset=tag_name, n_views=len(indices))
                    rows += new
                    preds = [r["pred_cm"] for r in new if r["pred_cm"] != ""]
                    print(f"{tag}: " + (", ".join(f"{x:.1f}cm" for x in preds) if preds else "无效"), flush=True)
                except KeyboardInterrupt:
                    common.write_csv(results, rows, FIELDS, key)
                    raise
                except Exception as error:
                    failed += 1
                    traceback.print_exc()
                    rows.append(common.failed_row(manifest, path, error, model=variant, subset=tag_name, n_views=len(indices)))
                    print(f"{tag}: 失败，继续下一个", flush=True)
                common.write_csv(results, rows, FIELDS, key)
        del model
    print(f"结果：{results}（失败 {failed} 个）\n")
    report(results)
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
