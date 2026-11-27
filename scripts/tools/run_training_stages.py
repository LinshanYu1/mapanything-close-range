"""先分别计时一轮，再训练 A30/B30，合并最佳模块参数后训练 AB10。"""
import argparse
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))


def merge_modules(a_dir, b_dir, output, weights_name='model_best.pt'):
    import torch
    from close_range.models.model_io import _validate_config
    states = []
    for variant, directory, prefix in [('A', a_dir, 'scale_correction.'),
                                        ('B', b_dir, 'info_sharing.view_scorer.')]:
        config = json.loads((directory / 'model_config.json').read_text())
        _validate_config(config)
        state = torch.load(directory / weights_name, map_location='cpu', weights_only=True)
        if config['variant'] != variant or state.get('model_config') != config or state.get('format') != 'close_range_model_v1':
            raise ValueError(f'{variant} 权重或配置不匹配')
        if not state['parameters'] or any(not k.startswith(prefix) for k in state['parameters']):
            raise ValueError(f'{variant} 参数名不正确')
        states.append(state)
    a, b = states
    for key in ('format_version', 'architecture', 'base_model', 'revision', 'feature_dim'):
        if a['model_config'][key] != b['model_config'][key]:
            raise ValueError(f'A/B 配置不兼容：{key}')
    config = dict(a['model_config'], variant='AB', b_hidden_dim=b['model_config']['b_hidden_dim'])
    _validate_config(config)
    output.mkdir(parents=True, exist_ok=False)
    parameters = {**a['parameters'], **b['parameters']}
    torch.save(dict(format='close_range_model_v1', model_config=config,
                    parameters=parameters, epoch=None, global_step=0), output / 'model.pt')
    (output / 'model_config.json').write_text(json.dumps(config, indent=2))
    (output / 'initialization.json').write_text(json.dumps({
        'A': str(a_dir.relative_to(ROOT)), 'B': str(b_dir.relative_to(ROOT)),
        'weights_name': weights_name,
    }, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--split', default='configs/local_mixed_split.json')
    parser.add_argument('--run-root', default='checkpoints/staged_' + datetime.now().strftime('%Y%m%d_%H%M%S'))
    parser.add_argument('--device', choices=['mps', 'cuda', 'cpu'], default='mps')
    parser.add_argument('--learning-rate', type=float, default=5e-4)
    parser.add_argument('--validation-interval', type=int, default=3)
    parser.add_argument('--seed', type=int, default=443)
    args = parser.parse_args()
    root = (ROOT / args.run_root).resolve()
    if not root.is_relative_to(ROOT):
        parser.error('run-root 必须在项目内')
    if root.exists():
        parser.error('run-root 已存在，请换一个目录')
    from close_range.data_split import load_split
    load_split(args.split)
    root.mkdir(parents=True)
    timings = []

    def run(stage, variant, epochs, init=None):
        output = root / stage / variant
        cmd = [sys.executable, '-u', str(ROOT / 'src/close_range/training/main.py'),
               '--variant', variant, '--split', args.split, '--output-dir', str(output),
               '--device', args.device, '--epochs', str(epochs), '--learning-rate', str(args.learning_rate),
               '--seed', str(args.seed), '--validation-interval', str(args.validation_interval)]
        if init is not None:
            cmd += ['--init-from', str(init), '--weights-name', 'model.pt']
        print(f'\n开始 {stage}/{variant}：{epochs} epoch', flush=True)
        start = time.perf_counter()
        result = subprocess.run(cmd, cwd=ROOT)
        elapsed = time.perf_counter() - start
        timings.append(dict(stage=stage, variant=variant, epochs=epochs,
                            elapsed_seconds=elapsed, returncode=result.returncode))
        (root / 'timings.json').write_text(json.dumps(timings, indent=2))
        print(f'{stage}/{variant} 总耗时（含加载和保存）：{elapsed / 60:.2f} 分钟', flush=True)
        if result.returncode:
            raise SystemExit(f'{stage}/{variant} 失败，已停止后续阶段；请查看上面的错误。')

    # 计时试跑不作为正式 A/B 的初始化，正式轮数为额外的30/30/10。
    for variant in ('A', 'B', 'AB'):
        run('benchmark', variant, 1)
    run('train', 'A', 30)
    run('train', 'B', 30)
    merged = root / 'AB_init'
    merge_modules(root / 'train/A', root / 'train/B', merged)
    run('train', 'AB', 10, merged)
    print(f'全部完成：{root}', flush=True)


if __name__ == '__main__':
    main()
