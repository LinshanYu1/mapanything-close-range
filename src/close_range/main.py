"""距离测试入口：修改 TEST_ARGS 后直接运行，不训练、不修改模型参数。"""
import sys
from datetime import datetime
from pathlib import Path

# 支持 PyCharm 直接运行，路径始终以项目目录为基准。
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from close_range.common import ROOT, REVISION, read, save, sha, portable_path

TEST_ARGS = [
    "B",                                      # 0: baseline / A / B / AB
    "checkpoints/graspnet_A",                  # 1: modelA_dir
    "checkpoints/graspnet_B",                  # 2: modelB_dir
    "checkpoints/graspnet_AB",                 # 3: modelAB_dir，联合训练后的权重
    "data/manifests/phone/scene-001.json",      # 4: 场景清单，包含图片和标注位置
    "model_best.pt",                          # 5: model_best.pt / model.pt
    "multi",                                  # 6: single / multi / both
    "mps",                                    # 7: mps / cuda / cpu
    "outputs/distance_tests",                 # 8: 输出根目录，每次自动新建子目录
]


def parse_settings(values):
    if len(values) != 9:
        raise ValueError("TEST_ARGS 需要 9 项，请按数组注释填写")
    variant, modelA_dir, modelB_dir, modelAB_dir, manifest, weights, mode, device, output = values
    if variant not in {"baseline", "A", "B", "AB"}:
        raise ValueError("模型必须为 baseline、A、B 或 AB")
    if mode not in {"single", "multi", "both"}:
        raise ValueError("输入模式必须为 single、multi 或 both")
    if device not in {"mps", "cuda", "cpu"}:
        raise ValueError("设备必须为 mps、cuda 或 cpu")
    if weights not in {"model_best.pt", "model.pt"}:
        raise ValueError("权重文件必须为 model_best.pt 或 model.pt")
    directory = {"A": modelA_dir, "B": modelB_dir, "AB": modelAB_dir}.get(variant)
    return dict(variant=variant, model_dir=None if directory is None else ROOT / directory,
                manifest=ROOT / manifest, weights=weights, mode=mode,
                device=device, output=ROOT / output)


def main(values=None):
    settings = parse_settings(TEST_ARGS if values is None else values)
    manifest = read(settings["manifest"])
    # 复用已验证的预测和距离评分流程，避免两套评分规则。
    sys.path.insert(0, str(ROOT / "scripts"))
    from run_baseline import prepare_inputs, predict_scene, load_model as load_baseline
    from evaluate import evaluate_run
    from close_range.models.model_io import load_model

    modes = ["single", "multi"] if settings["mode"] == "both" else [settings["mode"]]
    prepared = {mode: prepare_inputs(manifest, mode) for mode in modes}
    annotation = manifest.get("annotation")
    if not annotation or not (ROOT / annotation).is_file():
        raise FileNotFoundError("此场景缺少点对标注；请先标注 A/B 端点")
    variant = settings["variant"]
    provenance = None
    revision = REVISION
    if variant == "baseline":
        model = load_baseline(settings["device"])
    else:
        config = read(settings["model_dir"] / "model_config.json")
        if config["variant"] != variant:
            raise ValueError(f"选择了 {variant}，但目录中的模型类型是 {config['variant']}")
        weights = settings["model_dir"] / settings["weights"]
        provenance = dict(path=portable_path(weights), sha256=sha(weights))
        revision = config["revision"]
        model = load_model(settings["model_dir"], device=settings["device"],
                           weights_name=settings["weights"])
    output = settings["output"] / f"{variant}_{datetime.now():%Y%m%d_%H%M%S_%f}"
    rows = []
    for mode in modes:
        target = output / mode
        print(f"测试 {variant}，{mode}，{len(prepared[mode][0])} 张图片", flush=True)
        predict_scene(model, manifest, mode, settings["device"], target, prepared[mode])
        run = read(target / "run.json")
        run.update(variant=variant, module_weights=provenance, revision=revision)
        save(target / "run.json", run)
        result = evaluate_run(manifest, target)
        rows.append(dict(variant=variant, mode=mode, **result))
        save(output / "summary.json", rows)
        for pair in result["results"]:
            if pair["prediction_valid"]:
                print(f"{pair['pair_id']}: 真实 {pair['reference_cm']:.2f} cm；"
                      f"预测 {pair['predicted_cm']:.2f} cm；"
                      f"相对误差 {pair['relative_error_pct']:.2f}%", flush=True)
            else:
                print(f"{pair['pair_id']}: 无效预测，{pair.get('reason', '')}", flush=True)
    print(f"结果目录：{portable_path(output)}", flush=True)
    return rows


if __name__ == "__main__":
    main()
