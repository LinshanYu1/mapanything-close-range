"""训练循环、单/双组损失与 epoch 恢复测试，不下载预训练模型。"""
import copy
import json
import tempfile
import unittest
from pathlib import Path
import sys

import torch
from torch import nn
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from close_range.training.trainer import Trainer
from close_range.training.losses import distance_loss


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale_head = nn.Linear(4, 1)
        self.info_sharing = nn.Module()
        self.info_sharing.dim = 4
        self.restored = False

    def _configure_geometric_input_config(self, *args):
        self.restored = False

    def _restore_original_geometric_input_config(self):
        self.restored = True

    def forward(self, views, **kwargs):
        parameter = next(self.parameters())
        global_tokens = torch.ones(1, 1, 4, device=parameter.device)
        local_tokens = torch.arange(8., device=parameter.device).reshape(1, 2, 4)
        correction = self.scale_correction(global_tokens, local_tokens)
        xyz = torch.tensor([[[[0., 0., 0.], [.3, 0., 0.]]]], device=parameter.device)
        return [{'pts3d': xyz * correction.reshape(1, 1, 1, 1)}]


class TinyDataset:
    def __init__(self, training, two_views=True):
        self.training = training
        self.seed = 0
        self.use_consistency = two_views
        scene_id = 'train' if training else 'val'
        self.scenes = [dict(scene_id=scene_id, source='phone',
            views=[dict(id='ref', sha256=scene_id)], reference_index=0, model_size=(2, 1),
            pairs=[dict(id='p', points_xy=torch.tensor([[0.,0.],[1.,0.]]), distance_m=.4)])]
        self.sample = dict(scene_id=scene_id, source='phone',
            views_1=[{'img':torch.zeros(1,3,1,2)}],
            views_2=[{'img':torch.zeros(1,3,1,2)}] if two_views else None,
            points_xy=torch.tensor([[[0.,0.],[1.,0.]]]), target_distances_m=torch.tensor([.4]))

    def __len__(self): return 1
    def __getitem__(self, index): return self.sample
    def set_epoch(self, epoch): self.epoch = epoch


class TrainingTests(unittest.TestCase):
    def test_single_and_double_losses(self):
        d1 = torch.tensor([.3], requires_grad=True)
        truth = torch.tensor([.4])
        single = distance_loss(d1, truth)
        double = distance_loss(d1, truth, d1)
        torch.testing.assert_close(single['loss'], double['loss'])
        self.assertEqual(single['consistency'].item(), 0)
        single['loss'].backward()
        self.assertGreater(d1.grad.abs().item(), 0)
        with self.assertRaises(ValueError): distance_loss(d1, torch.tensor([0.]))
        with self.assertRaises(ValueError): distance_loss(d1, torch.tensor([.4,.5]))

    def test_update_save_resume_matches_uninterrupted(self):
        with tempfile.TemporaryDirectory() as tmp:
            train, val = TinyDataset(True, False), TinyDataset(False, False)
            original = TinyModel()
            full = Trainer(copy.deepcopy(original), Path(tmp)/'full', device='cpu')
            full.fit(train, val, epochs=2)
            part = Trainer(copy.deepcopy(original), Path(tmp)/'part', device='cpu')
            part.fit(train, val, epochs=1)
            resumed = Trainer(copy.deepcopy(original), Path(tmp)/'part', device='cpu')
            resumed.fit(train, val, epochs=2, resume=Path(tmp)/'part/checkpoints/latest.pt')
            self.assertEqual(resumed.global_step, 2)
            self.assertTrue(resumed.model.restored)
            self.assertTrue((Path(tmp)/'part/checkpoints/best.pt').is_file())
            exported = torch.load(Path(tmp)/'part/model.pt', weights_only=True)
            self.assertNotIn('optimizer', exported)
            self.assertEqual(exported['epoch'], 2)
            for name, parameter in resumed.model.named_parameters():
                if parameter.requires_grad:
                    torch.testing.assert_close(exported['parameters'][name], parameter)
            config = json.loads((Path(tmp)/'part/training_config.json').read_text())
            self.assertEqual(config['target_epochs'], 2)
            self.assertEqual(config['variant'], 'A')
            self.assertEqual(config['train_data']['scenes'][0]['scene_id'], 'train')
            self.assertTrue((Path(tmp)/'part/model_best.pt').is_file())
            for a,b in zip(full.parameters,resumed.parameters):
                torch.testing.assert_close(a,b,atol=0,rtol=0)
            self.assertTrue(all(float(s['step'])==2 for s in resumed.optimizer.state.values()))
            self.assertTrue(all(p.grad is None for p in resumed.model.parameters() if not p.requires_grad))

    def test_validation_failure_and_overlap(self):
        with tempfile.TemporaryDirectory() as tmp:
            trainer = Trainer(TinyModel(),Path(tmp)/'run',device='cpu')
            val = TinyDataset(False)
            val.sample['target_distances_m'][0] = float('nan')
            report = trainer.validate(val)
            self.assertEqual(report['summary']['invalid_scenes'],1)
            self.assertIsNone(report['summary']['relative_error'])
            train = TinyDataset(True)
            val = TinyDataset(False)
            val.scenes = train.scenes
            with self.assertRaisesRegex(ValueError,'重叠'):
                trainer.fit(train,val,1)

    def test_resume_rejects_changed_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            train,val = TinyDataset(True),TinyDataset(False)
            Trainer(TinyModel(),tmp,device='cpu').fit(train,val,1)
            train.seed = 10
            with self.assertRaisesRegex(ValueError,'不一致'):
                Trainer(TinyModel(),tmp,device='cpu').fit(train,val,2,resume=Path(tmp)/'checkpoints/latest.pt')


if __name__ == '__main__': unittest.main()
