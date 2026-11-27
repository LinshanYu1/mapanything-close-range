"""训练进程独享的冻结编码器缓存；不缓存可训练的融合网络。"""
import hashlib
import tempfile
from pathlib import Path
import torch


class FrozenEncoderCache:
    def __init__(self, model):
        self.encoder = model.encoder
        self.original = model._encode_n_views
        self.storage = tempfile.TemporaryDirectory(prefix='mapanything_encoder_')
        self.root = Path(self.storage.name)
        self.hits = self.misses = 0
        self.versions = self._versions()

    def _versions(self):
        return tuple((id(t), t._version) for t in
                     [*self.encoder.parameters(), *self.encoder.buffers()])

    def __call__(self, views):
        # 仅用于冻结、确定性的 eval 编码器。状态改变后禁止复用旧特征。
        if (self.encoder.training or any(p.requires_grad for p in self.encoder.parameters())
                or any(v['img'].requires_grad for v in views)):
            raise RuntimeError('编码器缓存要求冻结参数、eval模式及无输入梯度')
        if self._versions() != self.versions:
            raise RuntimeError('编码器参数或缓冲区已变化，请重建缓存')
        if torch.is_autocast_enabled(views[0]['img'].device.type):
            return self.original(views)
        paths = []
        for view in views:
            img = view['img'].detach().cpu().contiguous()
            key = hashlib.sha256()
            key.update(repr((tuple(img.shape), str(img.dtype), view['data_norm_type'])).encode())
            key.update(img.view(torch.uint8).numpy().tobytes())
            paths.append(self.root / (key.hexdigest() + '.pt'))
        # 相同输入只编码一次；缓存命中不再随其他缺失视角重复编码。
        missing = {}
        for path, view in zip(paths, views):
            if path.is_file():
                self.hits += 1
            else:
                missing.setdefault(path, view)
        if missing:
            with torch.no_grad():
                features, registers = self.original(list(missing.values()))
            self.misses += len(missing)
            for i, path in enumerate(missing):
                torch.save({
                    'features': features[i].detach().cpu().clone(),
                    'registers': None if registers is None else registers[i].detach().cpu().clone(),
                }, path)
            del features, registers

        # 必须恢复请求顺序（包括重复视角），不能按缓存顺序返回。
        records = [torch.load(path, map_location='cpu', weights_only=True) for path in paths]
        features = tuple(r['features'].to(v['img'].device) for r, v in zip(records, views))
        registers = None if records[0]['registers'] is None else tuple(
            r['registers'].to(v['img'].device) for r, v in zip(records, views))
        return features, registers

    def close(self):
        self.storage.cleanup()
