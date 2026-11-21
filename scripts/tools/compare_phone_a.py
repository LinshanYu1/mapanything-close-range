"""同一手机输入上对比原始 MapAnything 与训练后的 A；不使用真实尺度对齐。"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
from close_range.common import ROOT, REVISION, read, save, sha, portable_path
from close_range.models.model_io import load_module_parameters
from close_range.models.scale_correction import attach_scale_correction
from run_baseline import load_model, predict_scene, prepare_inputs
from evaluate import evaluate_run


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', default='data/manifests/phone/scene-001.json')
    p.add_argument('--model-dir', default='checkpoints/graspnet_A')
    p.add_argument('--weights-name', default='model_best.pt', choices=['model_best.pt', 'model.pt'])
    p.add_argument('--output', required=True)
    p.add_argument('--device', default='mps')
    a = p.parse_args()
    out = ROOT / a.output
    if out.exists():
        raise FileExistsError(out)
    directory = ROOT / a.model_dir
    config = read(directory / 'model_config.json')
    if config['variant'] != 'A' or config['revision'] != REVISION:
        raise ValueError('Requires A checkpoint using the same baseline revision')
    weights = directory / a.weights_name
    digest = sha(weights)
    manifest = read(ROOT / a.manifest)
    prepared = {mode: prepare_inputs(manifest, mode) for mode in ['single', 'multi']}
    model = load_model(a.device)
    rows = []
    for variant in ['baseline', 'A']:
        if variant == 'A':
            attach_scale_correction(model, hidden_dim=config['a_hidden_dim'])
            state = load_module_parameters(model, directory, a.weights_name)
            model.requires_grad_(False)
            model.eval()
        for mode in ['single', 'multi']:
            target = out / f'{variant}_{mode}'
            predict_scene(model, manifest, mode, a.device, target, prepared[mode])
            run = read(target / 'run.json')
            run.update(variant=variant, module_weights=None if variant == 'baseline' else
                       dict(path=portable_path(weights), sha256=digest, epoch=state.get('epoch')))
            save(target / 'run.json', run)
            result = evaluate_run(manifest, target)
            rows.append(dict(variant=variant, mode=mode, **result))
            save(out / 'comparison.json', rows)
            print(rows[-1], flush=True)


if __name__ == '__main__':
    main()
