"""Real-data B/A+B smoke training and checkpoint inference. Not a benchmark."""
import argparse
from copy import deepcopy
from pathlib import Path
import sys

import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from run_baseline import load_model, prepare_inputs, predict_scene
from evaluate import evaluate_run
from close_range.common import read, resolve, save, phone_distance
from close_range.models import attach_view_fusion, attach_scale_correction, distance_training_loss
from mapanything.utils.inference import validate_input_views_for_inference, preprocess_input_views_for_inference


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',default='data/manifests/phone/scene-001.json')
    parser.add_argument('--output',required=True)
    parser.add_argument('--device',choices=['cpu','mps','cuda'],default='mps')
    parser.add_argument('--steps',type=int,default=1)
    parser.add_argument('--lr',type=float,default=0.0005)
    parser.add_argument('--consistency-weight',type=float,default=0.1)
    parser.add_argument('--hidden-dim',type=int,default=128)
    parser.add_argument('--with-a',action='store_true',help='Train A and B jointly, A starts at identity')
    parser.add_argument('--checkpoint',help='Load our saved B or A+B checkpoint; use --steps 0 for inference')
    args=parser.parse_args()
    if args.steps<0 or args.lr<=0 or args.consistency_weight<0: parser.error('Invalid training settings')
    out=resolve(args.output)
    if out.exists(): parser.error('Choose a new output directory')
    manifest=read(resolve(args.manifest));ann=read(resolve(manifest['annotation']))
    ref=manifest['views'][manifest['reference_index']]
    if ann['reference_view_id']!=ref['id'] or ann['image_sha256']!=ref['sha256']:
        raise ValueError('Annotation/reference mismatch')
    pair=next(p for p in ann['pairs'] if p['label_valid'])
    if manifest['source']=='phone':
        distance,_=phone_distance(manifest['scene_id'],manifest.get('distance_config'))
        if abs(distance*.0254-pair['distance_m'])>1e-8: raise ValueError('Measurement changed after annotation')
    if args.steps and len(manifest['views'])<3:
        raise ValueError('Training requires a reference plus at least two other views')
    torch.manual_seed(0)
    model=load_model(args.device)
    saved=torch.load(resolve(args.checkpoint),map_location='cpu',weights_only=True) if args.checkpoint else None
    if saved and saved.get('format')!='close_range_b_v1': raise ValueError('Unsupported checkpoint')
    with_a=args.with_a or bool(saved and saved['a_state'] is not None)
    if saved and args.with_a and saved['a_state'] is None:
        raise ValueError('Cannot add A silently while restoring a B-only checkpoint')
    if with_a: attach_scale_correction(model,hidden_dim=saved['a_hidden_dim'] if saved else 128)
    attach_view_fusion(model,hidden_dim=saved['b_hidden_dim'] if saved else args.hidden_dim)
    if with_a: model.scale_correction.requires_grad_(True)
    if saved:
        model.info_sharing.view_scorer.load_state_dict(saved['b_state'],strict=True)
        if with_a: model.scale_correction.load_state_dict(saved['a_state'],strict=True)
    model.eval()  # Deterministic features; eval does not disable gradients.
    selected=[p for p in model.parameters() if p.requires_grad]
    optimizer=torch.optim.AdamW(selected,lr=args.lr,weight_decay=0)
    points=torch.tensor(pair['points_xy']).round().long()
    target=torch.tensor(pair['distance_m'],device=args.device)
    if not torch.isfinite(target) or target<=0: raise ValueError('Invalid reference distance')

    # Same reference, two distinct supporting views. Manifest contains processed pixels.
    others=[i for i in range(len(manifest['views'])) if i!=manifest['reference_index']]
    subsets=[]
    for other in others[:2]:
        sub=deepcopy(manifest);sub['views']=[ref,manifest['views'][other]];sub['reference_index']=0
        views,_,size=prepare_inputs(sub,'multi')
        if (points<0).any() or (points[:,0]>=size[0]).any() or (points[:,1]>=size[1]).any():
            raise ValueError('Endpoint outside processed image')
        views=validate_input_views_for_inference(views)
        for view in views:
            for key,value in view.items():
                if torch.is_tensor(value) and key!='true_shape': view[key]=value.to(args.device)
        subsets.append(preprocess_input_views_for_inference(views))

    rows=[]
    if args.steps:
        model._configure_geometric_input_config(False,False,False,False,False)
        def predict(views):
            xyz=model(views,memory_efficient_inference=True,minibatch_size=1)[0]['pts3d'][0]
            a,b=[xyz[y,x] for x,y in points.tolist()]
            return torch.linalg.vector_norm(a-b)
        try:
            with torch.no_grad():
                model.info_sharing.view_fusion_enabled=False
                baseline=[predict(v).item() for v in subsets]
                model.info_sharing.view_fusion_enabled=True
                initial=[predict(v).item() for v in subsets]
            if not saved:
                torch.testing.assert_close(torch.tensor(initial),torch.tensor(baseline),rtol=1e-4,atol=1e-6)
            for step in range(args.steps):
                optimizer.zero_grad(set_to_none=True)
                d1,d2=[predict(v) for v in subsets]
                loss,sup,cons=distance_training_loss(d1,d2,target,args.consistency_weight)
                loss.backward()
                grads=[p.grad for p in model.info_sharing.view_scorer.parameters() if p.grad is not None]
                if not grads or not all(torch.isfinite(g).all() for g in grads) or not any(g.abs().sum()>0 for g in grads):
                    raise AssertionError('B gradients missing, nonfinite, or all zero')
                if any(p.grad is not None for p in model.parameters() if not p.requires_grad):
                    raise AssertionError('Frozen backbone received gradients')
                norm=torch.nn.utils.clip_grad_norm_(selected,1.0)
                rows.append(dict(step=step,distance_m=[d1.item(),d2.item()],loss=loss.item(),
                                 supervised=sup.item(),consistency=cons.item(),gradient_norm=float(norm)))
                optimizer.step();print(rows[-1],flush=True)
            with torch.no_grad():
                final=[predict(v).item() for v in subsets]
            save(out/'training.json',dict(reference_m=target.item(),baseline_m=baseline,initial_m=initial,
                 final_m=final,rows=rows,seed=0,lr=args.lr,consistency_weight=args.consistency_weight,
                 subsets=[[ref['id'],manifest['views'][i]['id']] for i in others[:2]],
                 note='One-scene smoke training; does not establish accuracy or generalization.'))
        finally:
            model._restore_original_geometric_input_config()
            model.info_sharing.view_fusion_enabled=True
    out.mkdir(parents=True,exist_ok=True)
    checkpoint=dict(format='close_range_b_v1',b_hidden_dim=model.info_sharing.view_scorer.hidden_dim,
        b_state={k:v.detach().cpu() for k,v in model.info_sharing.view_scorer.state_dict().items()},
        a_hidden_dim=model.scale_correction.hidden_dim if with_a else None,
        a_state={k:v.detach().cpu() for k,v in model.scale_correction.state_dict().items()} if with_a else None,
        scene=manifest['scene_id'],steps=args.steps)
    torch.save(checkpoint,out/'modules.pt')
    # Exercise exact serialization path before final full-scene evaluation.
    restored=torch.load(out/'modules.pt',map_location='cpu',weights_only=True)
    model.info_sharing.view_scorer.load_state_dict(restored['b_state'])
    if with_a: model.scale_correction.load_state_dict(restored['a_state'])
    predict_scene(model,manifest,'multi',args.device,out/'prediction')
    metrics=evaluate_run(manifest,out/'prediction')
    print(f"Saved checkpoint and evaluated {metrics['eligible_pairs']} pairs in {out}",flush=True)


if __name__=='__main__': main()
