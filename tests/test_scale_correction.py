"""Checks for identity, view-major batching, gradients and scale integration."""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from close_range.models.scale_correction import GlobalLocalScaleCorrection, attach_scale_correction
from close_range.common import ROOT
from mapanything.models.mapanything.model import MapAnything


class ScaleCorrectionTests(unittest.TestCase):
    def test_identity_and_learning(self):
        torch.manual_seed(1)
        module = GlobalLocalScaleCorrection(8, 4)
        global_features = torch.randn(2, 1, 8)
        local_features = torch.randn(6, 7, 8)
        self.assertTrue(torch.equal(module(global_features, local_features), torch.ones(2, 1)))
        optimizer = torch.optim.SGD(module.parameters(), lr=0.05)
        loss = (module(global_features, local_features) - 0.5).square().mean()
        loss.backward()
        self.assertGreater(module.log_scale_head.weight.grad.abs().sum().item(), 0)
        optimizer.step()
        optimizer.zero_grad()
        result = module(global_features, local_features)
        self.assertLess((result - 0.5).square().mean().item(), loss.item())
        result.sum().backward()
        self.assertGreater(module.key.weight.grad.abs().sum().item(), 0)
        self.assertGreater(module.global_projection.weight.grad.abs().sum().item(), 0)

    def test_scenes_not_mixed_and_views_permutable(self):
        torch.manual_seed(2)
        module = GlobalLocalScaleCorrection(8, 4)
        nn.init.normal_(module.log_scale_head.weight)
        global_features = torch.randn(2, 1, 8)
        local_features = torch.randn(3, 2, 5, 8)
        combined = module(global_features, local_features.flatten(0, 1))
        separate = torch.cat([module(global_features[i:i+1], local_features[:, i]) for i in range(2)])
        torch.testing.assert_close(combined, separate)
        torch.testing.assert_close(combined, module(global_features, local_features.flip(0).flatten(0, 1)))

    def test_real_downstream_hook(self):
        class Head(nn.Module):
            def forward(self, value):
                return SimpleNamespace(decoded_channels=torch.full((2, 1, 1), 3.0))
        class Adaptor(nn.Module):
            def forward(self, value):
                return SimpleNamespace(value=value.adaptor_feature)
        fake = SimpleNamespace(device=torch.device('cpu'), pred_head_type='linear',
                               downstream_dense_head=lambda inputs, shape: inputs,
                               scale_head=Head(), scale_adaptor=Adaptor())
        local = torch.randn(6, 8, 2, 3)
        global_features = torch.randn(2, 8, 1)
        baseline = MapAnything.downstream_head(fake, local, global_features, (14, 14))[2]
        fake.scale_correction = GlobalLocalScaleCorrection(8, 4)
        corrected = MapAnything.downstream_head(fake, local, global_features, (14, 14))[2]
        torch.testing.assert_close(baseline, corrected, rtol=0, atol=0)
        with torch.no_grad():
            fake.scale_correction.log_scale_head.bias.fill_(torch.log(torch.tensor(2.0)))
        corrected = MapAnything.downstream_head(fake, local, global_features, (14, 14))[2]
        torch.testing.assert_close(corrected, baseline * 2)
        corrected.sum().backward()
        self.assertIsNotNone(fake.scale_correction.log_scale_head.bias.grad)

    def test_attach_freezes_only_backbone(self):
        model = nn.Module()
        model.scale_head = nn.Linear(8, 1)
        model.info_sharing = SimpleNamespace(dim=8)
        original = model.scale_head.weight.detach().clone()
        attach_scale_correction(model, hidden_dim=4)
        self.assertFalse(model.scale_head.weight.requires_grad)
        self.assertTrue(all(p.requires_grad for p in model.scale_correction.parameters()))
        self.assertTrue(torch.equal(original, model.scale_head.weight))
        with self.assertRaises(ValueError):
            attach_scale_correction(model)


if __name__ == '__main__':
    unittest.main()
