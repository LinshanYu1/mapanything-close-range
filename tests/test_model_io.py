"""模型目录独立加载、配置绑定和损坏参数检查。"""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import sys

import torch
from torch import nn
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from close_range.models import attach_scale_correction, attach_view_fusion
from close_range.models.model_io import save_model, load_model, get_model_config
from test_view_fusion import model as small_model, inputs, tensor


class ModelIOTests(unittest.TestCase):
    def test_all_variants_roundtrip_including_frozen_modules(self):
        for variant in ('A', 'B', 'AB'):
            with self.subTest(variant=variant), tempfile.TemporaryDirectory() as tmp:
                base = small_model(False)
                base.scale_head = nn.Linear(16,1)
                model = copy.deepcopy(base)
                if variant in ('A','AB'): attach_scale_correction(model,hidden_dim=8)
                if variant in ('B','AB'): attach_view_fusion(model,hidden_dim=8)
                # 非零模块参数，避免仅检查恒等初始化。
                with torch.no_grad():
                    if variant in ('A','AB'): model.scale_correction.log_scale_head.bias.fill_(.12)
                    if variant in ('B','AB'): model.info_sharing.view_scorer.net[-1].weight.normal_()
                x = inputs()
                before = tensor(model.info_sharing(x))
                model.requires_grad_(False)
                save_model(model,tmp,weights_name='model_best.pt')
                with patch('close_range.models.model_io._load_base_model',return_value=copy.deepcopy(base)):
                    restored=load_model(tmp,device='cpu')
                torch.testing.assert_close(tensor(restored.info_sharing(x)),before)
                if variant in ('A','AB'):
                    g,l=torch.randn(2,1,16),torch.randn(6,4,16)
                    torch.testing.assert_close(model.scale_correction(g,l),restored.scale_correction(g,l))
                self.assertFalse(restored.training)
                self.assertTrue(all(not p.requires_grad for p in restored.parameters()))
                # 推理模型也可重新保存，不能按 requires_grad 过滤为空。
                save_model(restored,tmp,weights_name='model.pt')
                self.assertTrue(torch.load(Path(tmp)/'model.pt',weights_only=True)['parameters'])

    def test_corruption_rejected(self):
        for corruption in ('missing','extra','shape','nan','config'):
            with self.subTest(corruption=corruption),tempfile.TemporaryDirectory() as tmp:
                base=small_model(False);base.scale_head=nn.Linear(16,1)
                model=copy.deepcopy(base);attach_scale_correction(model,8)
                save_model(model,tmp,weights_name='model_best.pt')
                path=Path(tmp)/'model_best.pt';state=torch.load(path,weights_only=True)
                name=next(iter(state['parameters']))
                if corruption=='missing':del state['parameters'][name]
                elif corruption=='extra':state['parameters']['unknown']=torch.zeros(1)
                elif corruption=='shape':state['parameters'][name]=torch.zeros(99)
                elif corruption=='nan':state['parameters'][name].fill_(float('nan'))
                else:state['model_config']['revision']='wrong-revision'
                torch.save(state,path)
                with patch('close_range.models.model_io._load_base_model',return_value=copy.deepcopy(base)):
                    with self.assertRaises(ValueError):load_model(tmp,device='cpu')


if __name__=='__main__':unittest.main()
