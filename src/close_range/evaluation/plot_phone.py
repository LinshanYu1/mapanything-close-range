"""手机结果的预测距离 vs 真实距离散点图。

  bash scripts/local.sh -m close_range.evaluation.plot_phone
  bash scripts/local.sh -m close_range.evaluation.plot_phone --results outputs/eval/phone_mixed/results.csv --out results/eval/phone_mixed_pred_vs_gt
"""
import argparse
import csv


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", default="results/eval/phone_results.csv")
    p.add_argument("--out", default="results/eval/phone_pred_vs_gt", help="输出文件名（不含扩展名）")
    a = p.parse_args(argv)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = [r for r in csv.DictReader(open(a.results))
            if r["status"] == "ok" and r["valid"] == "True"]
    n = len({r["scene"] for r in csv.DictReader(open(a.results))})
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), sharey=True)
    style = {"AB": ("#2ca02c", "D", "AB (both)", 20),
             "B": ("#d62728", "^", "B (view fusion)", 22),
             "A": ("#1f77b4", "s", "A (scale correction)", 22),
             "baseline": ("#555555", "o", "M0 (pretrained)", 55)}
    for ax, mode in zip(axes, ["single", "multi"]):
        for m, (c, mk, label, size) in style.items():
            pts = [(float(r["gt_cm"]), float(r["pred_cm"])) for r in rows if r["model"] == m and r["mode"] == mode]
                if not pts:
                    continue
            if m == "baseline":
                ax.scatter(*zip(*pts), s=size, facecolors="none", edgecolors=c, linewidths=1.1, marker=mk, label=label, zorder=3)
            else:
                ax.scatter(*zip(*pts), s=size, c=c, marker=mk, alpha=0.75, label=label, zorder=2)
        ax.plot([3, 300], [3, 300], "k--", lw=1, label="y = x")
        ax.fill_between([3, 300], [2.7, 270], [3.3, 330], color="k", alpha=0.07, label="±10%")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlim(4, 150); ax.set_ylim(4, 250)
        ax.set_xlabel("Ground-truth distance (cm)")
        ax.grid(True, which="both", alpha=0.2)
    axes[0].set_title("single-view input (B = M0 with one view)")
    axes[1].set_title("multi-view input")
    axes[0].set_ylabel("Predicted distance (cm)")
    h, l = axes[1].get_legend_handles_labels()
    wanted = ["M0 (pretrained)", "A (scale correction)", "B (view fusion)", "AB (both)", "y = x", "±10%"]
    order = [l.index(x) for x in wanted if x in l]
    axes[1].legend([h[i] for i in order], [l[i] for i in order], loc="upper left", fontsize=8)
    fig.suptitle(f"Phone scenes (n = {n})", y=1.0, fontsize=10)
    fig.tight_layout()
    fig.savefig(a.out + ".pdf"); fig.savefig(a.out + ".png", dpi=200)
    print(f"saved {a.out}.pdf / .png")


if __name__ == "__main__":
    main()
