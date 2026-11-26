"""汇总 close_range.evaluation.run 的 results.csv：总体和按距离分段的误差统计。

  bash scripts/local.sh -m close_range.evaluation.summarize outputs/eval/phone/results.csv
  bash scripts/local.sh -m close_range.evaluation.summarize outputs/eval/*/results.csv --latex

统计单位是点对（手机每个场景 1 对，GraspNet 每个场景通常 10 对）。
无效预测计入"无效"并算作失败（不在 10%/20% 以内），中位误差只统计有效预测。
"""
import argparse
import csv
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ORDER = {"baseline": 0, "A": 1, "B": 2, "AB": 3}
BINS = [("<20cm", 0, 20), ("20-60cm", 20, 60), (">=60cm", 60, float("inf"))]
COLS = ["source", "model", "mode", "bin", "pairs", "invalid", "median_rel_err_pct",
        "mean_rel_err_pct", "within_10pct", "within_20pct", "failed_scenes"]


def load(paths):
    rows = []
    for path in paths:
        with open(path, newline="") as f:
            rows += list(csv.DictReader(f))
    if not rows:
        raise SystemExit("results.csv 是空的")
    return rows


def stats(group):
    ok = [r for r in group if r["status"] == "ok"]
    errs = [float(r["rel_error_pct"]) for r in ok if r["valid"] == "True"]
    n = len(ok)
    pct = lambda k: 100 * sum(e <= k for e in errs) / n if n else None
    return dict(pairs=n, invalid=n - len(errs),
                median_rel_err_pct=statistics.median(errs) if errs else None,
                mean_rel_err_pct=statistics.mean(errs) if errs else None,
                within_10pct=pct(10), within_20pct=pct(20),
                failed_scenes=len({r["scene"] for r in group if r["status"] == "failed"}))


def table(rows):
    groups = defaultdict(list)
    for r in rows:
        key = (r["source"], r["model"], r["mode"])
        groups[key + ("all",)].append(r)
        if r["status"] == "ok":
            gt = float(r["gt_cm"])
            for name, lo, hi in BINS:
                if lo <= gt < hi:
                    groups[key + (name,)].append(r)
    bin_order = {"all": 0, **{b[0]: i + 1 for i, b in enumerate(BINS)}}
    out = []
    for key in sorted(groups, key=lambda k: (k[0], bin_order[k[3]], ORDER.get(k[1], 9), k[2])):
        out.append(dict(zip(["source", "model", "mode", "bin"], key), **stats(groups[key])))
    return out


def fmt(v):
    return "-" if v is None else (f"{v:.1f}" if isinstance(v, float) else str(v))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("results", nargs="+")
    p.add_argument("--latex", action="store_true", help="另外输出 LaTeX 表格行")
    a = p.parse_args(argv)
    summary = table(load(a.results))
    out = Path(a.results[0]).with_name("summary.csv")
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        w.writerows({k: fmt(v) if isinstance(v, float) else v for k, v in r.items()} for r in summary)
    head = f"{'source':9}{'bin':9}{'model':9}{'mode':7}{'pairs':>6}{'invalid':>8}{'median%':>9}{'<=10%':>7}{'<=20%':>7}"
    print(head)
    print("-" * len(head))
    last = None
    for r in summary:
        if last and (r["source"], r["bin"]) != last:
            print()
        last = (r["source"], r["bin"])
        print(f"{r['source']:9}{r['bin']:9}{r['model']:9}{r['mode']:7}{r['pairs']:>6}{r['invalid']:>8}"
              f"{fmt(r['median_rel_err_pct']):>9}{fmt(r['within_10pct']):>7}{fmt(r['within_20pct']):>7}")
    if a.latex:
        print("\n% source & bin & model & mode & pairs & median rel. err (%) & within 10% & within 20%")
        for r in summary:
            print(f"{r['source']} & {r['bin']} & {r['model']} & {r['mode']} & {r['pairs']} & "
                  f"{fmt(r['median_rel_err_pct'])} & {fmt(r['within_10pct'])} & {fmt(r['within_20pct'])} \\\\")
    print(f"\n已保存 {out}")
    if any(r["failed_scenes"] for r in summary):
        print("注意：有失败的场景，见 results.csv 中 status=failed 的行", file=sys.stderr)


if __name__ == "__main__":
    main()
