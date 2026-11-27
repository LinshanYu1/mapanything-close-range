"""对比真实冻结编码器的直接计算、增量缓存及热缓存。不会训练或更新权重。"""
import argparse
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
import torch
from close_range.common import ROOT,REVISION
from close_range.training.dataset import SceneDistanceDataset
from close_range.training.encoder_cache import FrozenEncoderCache
from mapanything.models import MapAnything


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',default='data/manifests/phone/scene-041.json')
    p.add_argument('--device',default='mps',choices=['cpu','mps','cuda'])
    a=p.parse_args()
    ds=SceneDistanceDataset(ROOT,[a.manifest],training=False,use_consistency=False)
    views=ds[0]['view_groups'][0]
    if len(views)<3:raise ValueError('请选至少3张图的场景')
    views=[{k:v.to(a.device) if torch.is_tensor(v) and k!='true_shape' else v for k,v in x.items()} for x in views]
    groups=[[views[0],views[1]],[views[0],views[2]],[views[2],views[0],views[1]]]
    model=MapAnything.from_pretrained('facebook/map-anything',revision=REVISION).to(a.device).eval().requires_grad_(False)
    cache=FrozenEncoderCache(model)
    def sync():
        if a.device=='mps':torch.mps.synchronize()
        elif a.device=='cuda':torch.cuda.synchronize()
    def run(fn):
        sync();start=time.perf_counter();out=[]
        with torch.no_grad():
            for group in groups:
                features,registers=fn(group)
                out.append((tuple(x.cpu() for x in features),None if registers is None else tuple(x.cpu() for x in registers)))
        sync();return out,time.perf_counter()-start
    try:
        with torch.no_grad():model._encode_n_views(groups[0])
        direct,t0=run(model._encode_n_views)
        incremental,t1=run(cache)
        hot,t2=run(cache)
        max_error=0.
        for actual in (incremental,hot):
            for x,y in zip(actual,direct):
                for xs,ys in zip(x,y):
                    if ys is None:
                        assert xs is None;continue
                    for xx,yy in zip(xs,ys):
                        max_error=max(max_error,(xx-yy).abs().max().item())
                        torch.testing.assert_close(xx,yy,rtol=1e-4,atol=1e-5)
        print(f'PASS 最大绝对误差={max_error:.8g}')
        print(f'直接编码={t0:.3f}s，首次增量缓存={t1:.3f}s，热缓存={t2:.3f}s')
        print(f'实际编码图片数={cache.misses}（应为3），缓存命中={cache.hits}')
        print('仅测编码阶段；不代表整轮训练的加速倍数。')
    finally:cache.close()


if __name__=='__main__':main()
