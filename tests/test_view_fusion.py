"""B tests against real, small UniCeption networks (no pretrained download)."""
import copy
import unittest
from pathlib import Path
import sys
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from close_range.models.view_fusion import (
    attach_view_fusion, distance_training_loss,
)
from uniception.models.info_sharing.alternating_attention_transformer import MultiViewAlternatingAttentionTransformerIFR
from uniception.models.info_sharing.base import MultiViewTransformerInput


def model(fused=True):
    result = nn.Module()
    result.info_sharing = MultiViewAlternatingAttentionTransformerIFR(
        name='b_test', input_embed_dim=16, dim=16, depth=4, num_heads=4,
        indices=[1], distinguish_ref_and_non_ref_views=True)
    for block in result.info_sharing.self_attention_blocks:
        block.attn.fused_attn = fused
    return result.eval()


def inputs(batch=2, views=3, registers=2):
    return MultiViewTransformerInput(
        features=[torch.randn(batch, 16, 2, 2) for _ in range(views)],
        additional_input_tokens_per_view=[torch.randn(batch, 16, registers) for _ in range(views)],
        additional_input_tokens=torch.randn(batch, 16, 1))


def tensor(output):
    return torch.cat(output[0].features, dim=0)


class ViewFusionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)

    def test_zero_identity_both_attention_paths_and_state_names(self):
        for fused in (False, True):
            m = model(fused); x = inputs()
            old = {k: v.clone() for k, v in m.state_dict().items()}
            expected = tensor(m.info_sharing(x))
            attach_view_fusion(m, hidden_dim=8)
            torch.testing.assert_close(tensor(m.info_sharing(x)), expected, atol=1e-6, rtol=1e-5)
            for key, value in old.items():
                torch.testing.assert_close(m.state_dict()[key], value, atol=0, rtol=0)
            self.assertEqual(type(m.info_sharing), MultiViewAlternatingAttentionTransformerIFR)
            self.assertTrue(all(p.requires_grad == ('view_scorer' in n)
                                for n,p in m.named_parameters()))
            with self.assertRaises(ValueError): attach_view_fusion(m)

    def test_explicit_bias_layout_and_global_layers_only(self):
        m = model(False)
        attach_view_fusion(m, 8)
        nn.init.normal_(m.info_sharing.view_scorer.net[-1].weight)
        captured = {}
        handles = []
        for i, block in enumerate(m.info_sharing.self_attention_blocks):
            def hook(module, args, kwargs, index=i):
                captured[index] = kwargs.get("attention_bias")
            handles.append(block.attn.register_forward_pre_hook(hook, with_kwargs=True))
        try:
            m.info_sharing(inputs(batch=2, views=3, registers=2))
        finally:
            for handle in handles:
                handle.remove()
        for i in (1, 3):
            self.assertIsNone(captured[i])
        for i in (0, 2):
            bias = captured[i]
            self.assertEqual(bias.shape, (2, 1, 1, 19))
            self.assertTrue(torch.equal(bias[..., -1], torch.zeros_like(bias[..., -1])))
            per_view = bias[:, 0, 0, :18].reshape(2, 3, 6)
            torch.testing.assert_close(per_view, per_view[:, :, :1].expand_as(per_view))
            torch.testing.assert_close(per_view[:, :, 0].sum(1), torch.zeros(2), atol=1e-6, rtol=0)

    def test_gradients_effect_checkpoint_and_fused_parity(self):
        m = model(False); attach_view_fusion(m, 8); x = inputs()
        before = tensor(m.info_sharing(x)).detach()
        optimizer = torch.optim.Adam(m.info_sharing.view_scorer.parameters(), lr=.03)
        loss = (tensor(m.info_sharing(x)) - torch.randn_like(before)).square().mean()
        loss.backward()
        scorer = m.info_sharing.view_scorer
        self.assertGreater(scorer.net[-1].weight.grad.abs().sum(), 0)
        self.assertTrue(all(p.grad is None for n,p in m.named_parameters() if 'view_scorer' not in n))
        optimizer.step(); optimizer.zero_grad()
        after = tensor(m.info_sharing(x))
        self.assertFalse(torch.allclose(before, after))
        after.square().mean().backward()
        self.assertGreater(scorer.net[1].weight.grad.abs().sum(), 0)
        m2 = model(True); attach_view_fusion(m2, 8)
        m2.load_state_dict(m.state_dict(), strict=True)
        torch.testing.assert_close(tensor(m2.info_sharing(x)), after, atol=2e-6, rtol=2e-5)
        m.info_sharing.view_fusion_enabled = False
        torch.testing.assert_close(tensor(m.info_sharing(x)), before)

    def test_batch_isolation_permutation_and_variable_views(self):
        m=model();attach_view_fusion(m,8)
        nn.init.normal_(m.info_sharing.view_scorer.net[-1].weight)
        for views in (1,2,5):
            x=inputs(2,views); combined=tensor(m.info_sharing(x)).reshape(views,2,16,2,2)
            for b in range(2):
                one=MultiViewTransformerInput(features=[f[b:b+1] for f in x.features],
                    additional_input_tokens_per_view=[r[b:b+1] for r in x.additional_input_tokens_per_view],
                    additional_input_tokens=x.additional_input_tokens[b:b+1])
                torch.testing.assert_close(tensor(m.info_sharing(one)),combined[:,b],atol=2e-6,rtol=2e-5)
            order=[0]+list(range(views-1,0,-1))
            perm=MultiViewTransformerInput(features=[x.features[i] for i in order],
                additional_input_tokens_per_view=[x.additional_input_tokens_per_view[i] for i in order],
                additional_input_tokens=x.additional_input_tokens)
            torch.testing.assert_close(tensor(m.info_sharing(perm)).reshape_as(combined),combined[order],atol=2e-6,rtol=2e-5)

    def test_consistency_alone_does_not_reward_accuracy(self):
        d=torch.tensor(.8,requires_grad=True);target=torch.tensor(.4)
        loss,supervised,consistency=distance_training_loss(d,d,target)
        self.assertEqual(consistency.item(),0)
        self.assertGreater(supervised.item(),0)
        loss.backward();self.assertGreater(d.grad.item(),0)
        with self.assertRaises(ValueError): distance_training_loss(d,d,torch.tensor(0.))


if __name__=='__main__': unittest.main()
