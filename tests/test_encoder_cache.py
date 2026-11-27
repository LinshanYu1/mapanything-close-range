"""CPU regression checks; no pretrained downloads required."""
import unittest
import torch
from close_range.training.encoder_cache import FrozenEncoderCache


class EncoderModel:
    def __init__(self):
        torch.manual_seed(443)
        self.encoder = torch.nn.Conv2d(3, 4, 1).double().eval().requires_grad_(False)
        self.encoded = 0

    def _encode_n_views(self, views):
        self.encoded += len(views)
        y = self.encoder(torch.cat([v['img'] for v in views]))
        return y.chunk(len(views)), y.mean((2, 3), keepdim=False).unsqueeze(-1).chunk(len(views))


class CacheTests(unittest.TestCase):
    def test_partial_hits_order_outputs_and_gradients(self):
        model = EncoderModel()
        baseline = EncoderModel()
        cache = FrozenEncoderCache(model)
        self.addCleanup(cache.close)
        views = [dict(img=torch.randn(1,3,5,5,dtype=torch.float64),data_norm_type=['test']) for _ in range(4)]
        for indices, encoded in [([0,1,2],3),([0,1,3],4),([0,2,3],4),([3,0,3],4)]:
            group = [views[i] for i in indices]
            actual = cache(group)
            expected = baseline._encode_n_views(group)
            self.assertEqual(model.encoded,encoded)
            for xs, ys in zip(actual,expected):
                for x,y in zip(xs,ys):
                    torch.testing.assert_close(x,y,rtol=1e-10,atol=1e-12)
            # 模拟下游可训练参数，验证缓存后梯度仍然一致。
            grads=[]
            for features in (actual[0],expected[0]):
                weight=torch.tensor(0.7,dtype=torch.float64,requires_grad=True)
                loss=sum((x*weight).square().mean() for x in features)
                loss.backward();grads.append(weight.grad)
            torch.testing.assert_close(*grads,rtol=1e-10,atol=1e-12)
        views[0]['img'].add_(1)
        cache([views[0]])
        self.assertEqual(model.encoded,5)
        with torch.no_grad():model.encoder.weight.add_(1)
        with self.assertRaisesRegex(RuntimeError,'已变化'):cache([views[1]])


if __name__=='__main__':unittest.main()
