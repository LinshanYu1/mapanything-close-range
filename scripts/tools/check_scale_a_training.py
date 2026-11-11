"""One-scene Module A training smoke test; not a generalization experiment."""
import argparse
import sys
from datetime import datetime
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from run_baseline import load_model, prepare_inputs
from close_range.common import read, resolve, save, phone_distance
from close_range.models import attach_scale_correction
from mapanything.utils.inference import (
    validate_input_views_for_inference,
    preprocess_input_views_for_inference,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', default='data/manifests/phone/scene-001.json')
    parser.add_argument('--steps', type=int, default=5)
    parser.add_argument('--device', default='mps')
    parser.add_argument('--output')
    args = parser.parse_args()
    if args.steps < 1:
        parser.error('--steps must be positive')
    output = resolve(args.output or f'outputs/scale_a_smoke_{datetime.now():%Y%m%d_%H%M%S}')
    if output.exists():
        parser.error('Output already exists')
    torch.manual_seed(0)
    manifest = read(resolve(args.manifest))
    annotation = read(resolve(manifest['annotation']))
    reference = manifest['views'][manifest['reference_index']]
    assert annotation['image_sha256'] == reference['sha256']
    assert annotation['reference_view_id'] == reference['id']
    pair = next(p for p in annotation['pairs'] if p['label_valid'])
    if manifest['source'] == 'phone':
        measured, _ = phone_distance(manifest['scene_id'], manifest.get('distance_config'))
        assert abs(measured * 0.0254 - pair['distance_m']) < 1e-8
    model = load_model(args.device)
    views, _, _ = prepare_inputs(manifest, 'single')
    views = validate_input_views_for_inference(views)
    for view in views:
        for key, value in view.items():
            if torch.is_tensor(value) and key not in {'true_shape'}:
                view[key] = value.to(args.device)
    views = preprocess_input_views_for_inference(views)
    model._configure_geometric_input_config(False, False, False, False, False)
    points = torch.tensor(pair['points_xy']).round().long()
    target = torch.tensor(pair['distance_m'], device=args.device)

    def forward_distance():
        prediction = model(views, memory_efficient_inference=True, minibatch_size=1)[0]
        xyz = prediction['pts3d'][0]
        a, b = [xyz[y, x] for x, y in points.tolist()]
        distance = torch.linalg.vector_norm(a - b)
        if not torch.isfinite(distance) or distance <= 0:
            raise ValueError('Invalid predicted endpoint distance')
        return distance

    with torch.no_grad():
        baseline = forward_distance().item()
    attach_scale_correction(model)
    model.eval()
    optimizer = torch.optim.AdamW(model.scale_correction.parameters(), lr=0.0005, weight_decay=0)
    rows = []
    for step in range(args.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        distance = forward_distance()
        loss = (distance.log() - target.log()).square()
        row = dict(step=step, distance_cm=distance.item()*100,
                   correction=distance.item()/baseline, loss=loss.item())
        if step == 0:
            assert abs(distance.item() - baseline) < 1e-6, 'Identity check failed'
        if step < args.steps:
            loss.backward()
            grads = [p.grad for p in model.scale_correction.parameters() if p.grad is not None]
            assert grads and all(torch.isfinite(g).all() for g in grads)
            assert any(g.abs().sum() > 0 for g in grads)
            assert all(p.grad is None for name,p in model.named_parameters() if not name.startswith('scale_correction.'))
            row['gradient_norm'] = float(torch.nn.utils.clip_grad_norm_(model.scale_correction.parameters(), 1.0))
            optimizer.step()
        rows.append(row)
        print(row, flush=True)
        save(output/'result.json', dict(scene=manifest['scene_id'], target_cm=target.item()*100,
             baseline_cm=baseline*100, rows=rows, note='Same-scene fitting smoke test, not held-out accuracy. Single RGB view, first labeled pair.'))
    assert rows[-1]['loss'] < rows[0]['loss'], 'Loss did not decrease'
    model._restore_original_geometric_input_config()
    print(f'PASS: identity, finite nonzero gradients, frozen backbone and decreasing loss. Report: {output / "result.json"}')


if __name__ == '__main__':
    main()
