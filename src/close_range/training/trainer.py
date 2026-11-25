"""逐场景训练 A / B / A+B；验证与 epoch 边界断点恢复。"""
import hashlib
import json
import math
import random
from pathlib import Path

import torch

from close_range.common import REVISION, portable_path
from close_range.models import attach_scale_correction, attach_view_fusion
from .checkpoint import (
    load_checkpoint, save_checkpoint, save_training_config,
)
from .losses import distance_loss
from close_range.models.model_io import get_model_config, save_model


class Trainer:
    def __init__(self, model, output_dir, variant="A", device="mps",
                 learning_rate=5e-4, consistency_weight=0.1, seed=0,
                 pretrained_revision=REVISION):
        if variant not in {"A", "B", "AB"}:
            raise ValueError("目前支持 A、B、AB；M1 需要单独明确解冻范围")
        if not math.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("学习率必须为有限正数")
        if not math.isfinite(consistency_weight) or consistency_weight < 0:
            raise ValueError("一致性权重必须为有限非负数")
        torch.manual_seed(seed)
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.variant = variant
        self.seed = seed
        self.output_dir = Path(output_dir)
        self.consistency_weight = consistency_weight
        if variant in {"A", "AB"}:
            attach_scale_correction(self.model)
        if variant in {"B", "AB"}:
            attach_view_fusion(self.model)
        if variant == "AB":
            self.model.scale_correction.requires_grad_(True)
        self.parameters = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            self.parameters, lr=learning_rate, weight_decay=0
        )
        # 与已验证的训练方式一致：关闭 dropout，但不关闭 autograd。
        self.model.eval()
        self.global_step = 0
        self.best_error = float("inf")
        self.config = dict(variant=variant, learning_rate=learning_rate,
                           consistency_weight=consistency_weight, seed=seed,
                           pretrained_revision=pretrained_revision)

    @classmethod
    def from_pretrained(cls, output_dir, **kwargs):
        from mapanything.models import MapAnything
        revision = kwargs.get("pretrained_revision", REVISION)
        model = MapAnything.from_pretrained("facebook/map-anything", revision=revision)
        return cls(model, output_dir, **kwargs)

    def move_views_to_device(self, views):
        # 不修改 Dataset 返回的字典；元数据 true_shape 保留在 CPU。
        return [
            {key: value.to(self.device) if torch.is_tensor(value) and key != "true_shape"
             else value for key, value in view.items()}
            for view in views
        ]

    def predict_distances(self, views, points_xy):
        predictions = self.model(
            self.move_views_to_device(views),
            memory_efficient_inference=True, minibatch_size=1,
        )
        xyz = predictions[0]["pts3d"][0]
        points = points_xy.to(xyz.device).round().long()
        a = xyz[points[:, 0, 1], points[:, 0, 0]]
        b = xyz[points[:, 1, 1], points[:, 1, 0]]
        return torch.linalg.vector_norm(a - b, dim=-1)

    def compute_loss(self, sample):
        target = sample["target_distances_m"].to(self.device)

        # 每个子集得到一组预测距离。
        # 堆叠后形状：[子集数量, 标注点对数量]
        distances = torch.stack([
            self.predict_distances(views, sample["points_xy"])
            for views in sample["view_groups"]
        ])

        losses = distance_loss(
            distances,
            target,
            consistency_weight=self.consistency_weight,
        )

        # 指标不参与反向传播。
        predictions = distances.detach()
        errors = (predictions - target).abs()

        # 所有不同子集之间的距离差，取平均，单位为 cm。
        num_groups = predictions.shape[0]
        subset_gap_cm = None

        if num_groups > 1:
            i, j = torch.triu_indices(
                num_groups,
                num_groups,
                offset=1,
                device=predictions.device,
            )
            subset_gap_cm = (
                    (predictions[i] - predictions[j]).abs().mean().item()
                    * 100
            )

        metrics = {
            key: value.detach().item()
            for key, value in losses.items()
        }

        metrics.update(
            mae_cm=errors.mean().item() * 100,
            relative_error=(errors / target).mean().item(),
            subset_gap_cm=subset_gap_cm,
        )

        return losses["loss"], metrics

    def train_step(self, sample):
        self.optimizer.zero_grad(set_to_none=True)
        loss, metrics = self.compute_loss(sample)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(
            self.parameters, max_norm=1.0, error_if_nonfinite=True
        )
        if not any(p.grad is not None for p in self.parameters):
            raise RuntimeError("可训练参数没有梯度")
        self.optimizer.step()
        self.global_step += 1
        metrics["gradient_norm"] = float(norm)
        return metrics

    @staticmethod
    def summarize(rows):
        valid = [row for row in rows if row.get("status") == "ok"]
        result = {"scenes": len(rows), "valid_scenes": len(valid),
                  "invalid_scenes": sum(r["status"] == "invalid" for r in rows),
                  "skipped_scenes": sum(r["status"] == "skipped" for r in rows)}
        for key in ("loss", "supervised", "consistency", "mae_cm", "relative_error", "subset_gap_cm"):
            values = [r[key] for r in valid if r.get(key) is not None]
            result[key] = sum(values) / len(values) if values else None
        return result

    def train_one_epoch(self, dataset, epoch):
        dataset.set_epoch(epoch)
        indices = list(range(len(dataset)))
        random.Random(self.seed + epoch).shuffle(indices)
        rows = []
        for index in indices:
            scene = dataset.scenes[index]
            row = {"scene_id": scene["scene_id"], "source": scene["source"]}
            if self.variant == "B" and len(scene["views"]) == 1:
                rows.append(dict(row, status="skipped", reason="单视角不能训练 B"))
                continue
            sample = dataset[index]
            rows.append(dict(row, status="ok", **self.train_step(sample)))
        if not any(row["status"] == "ok" for row in rows):
            raise ValueError("没有可训练的场景；B 至少需要双视角输入")
        return {"summary": self.summarize(rows), "scenes": rows}

    @torch.no_grad()
    def validate(self, dataset):
        rows = []
        for index in range(len(dataset)):
            sample = dataset[index]
            row = {"scene_id": sample["scene_id"], "source": sample["source"]}
            # 数据错误直接抛出；无效距离记录为失败，不悄悄排除。
            try:
                _, metrics = self.compute_loss(sample)
            except ValueError as error:
                rows.append(dict(row, status="invalid", reason=str(error)))
            else:
                rows.append(dict(row, status="ok", **metrics))
        return {
            "summary": self.summarize(rows),
            "by_source": {
                source: self.summarize([r for r in rows if r["source"] == source])
                for source in sorted({r["source"] for r in rows})
            },
            "scenes": rows,
        }

    @staticmethod
    def dataset_signature(dataset):
        # 包含图片哈希、标注和采样配置；恢复训练时防止静默更换数据。
        data = [{"scene_id": s["scene_id"], "source": s["source"],
                 "views": [(v["id"], v["sha256"]) for v in s["views"]],
                 "reference_index": s["reference_index"], "model_size": s["model_size"],
                 "pairs": [(p["id"], p["points_xy"].tolist(), p["distance_m"]) for p in s["pairs"]]}
                for s in dataset.scenes]
        payload = dict(scenes=data, seed=dataset.seed, training=dataset.training,
                       use_consistency=dataset.use_consistency)
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

    @staticmethod
    def describe_dataset(dataset):
        return {
            "seed": dataset.seed,
            "training": dataset.training,
            "use_consistency": dataset.use_consistency,
            "scenes": [
                {"scene_id": s["scene_id"], "source": s["source"],
                 "model_size": s["model_size"], "reference_index": s["reference_index"],
                 "views": s["views"]}
                for s in dataset.scenes
            ],
        }

    def fit(self, train_dataset, val_dataset, epochs, resume=None):
        if epochs < 1:
            raise ValueError("epochs 必须为正数")
        if not train_dataset.training or val_dataset.training:
            raise ValueError("训练 Dataset 应为 training=True，验证 Dataset 应为 False")
        train_ids = {(s["source"], s["scene_id"]) for s in train_dataset.scenes}
        val_ids = {(s["source"], s["scene_id"]) for s in val_dataset.scenes}
        train_hashes = {v["sha256"] for s in train_dataset.scenes for v in s["views"]}
        val_hashes = {v["sha256"] for s in val_dataset.scenes for v in s["views"]}
        if train_ids & val_ids or train_hashes & val_hashes:
            raise ValueError("训练和验证场景/图片重叠；同一布置的分组还需在划分清单时保证")
        self.config.update(train_signature=self.dataset_signature(train_dataset),
                           val_signature=self.dataset_signature(val_dataset))
        start_epoch = 0
        if resume is not None:
            state = load_checkpoint(resume, self.model, self.optimizer, self.config)
            start_epoch = state["next_epoch"]
            self.global_step = state["global_step"]
            self.best_error = state["best_error"]
            if start_epoch >= epochs:
                raise ValueError("epochs 是总轮数，必须大于 checkpoint 已完成轮数")
        elif self.output_dir.exists() and any(self.output_dir.iterdir()):
            raise ValueError("输出目录非空；请使用新目录或指定 resume")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        readable_config = dict(
            self.config,
            target_epochs=epochs,
            device=str(self.device),
            optimizer="AdamW",
            weight_decay=0,
            gradient_clip_norm=1.0,
            scene_batch_size=1,
            model_mode="eval_with_gradients",
            pretrained_model="facebook/map-anything",
            module_hidden_dim=128,
            resume_from=None if resume is None else portable_path(resume),
            train_data=self.describe_dataset(train_dataset),
            validation_data=self.describe_dataset(val_dataset),
        )
        save_training_config(self.output_dir / "training_config.json", readable_config)
        model_config = get_model_config(self.model, self.config["pretrained_revision"])
        (self.output_dir / "metrics").mkdir(exist_ok=True)
        self.model._configure_geometric_input_config(False, False, False, False, False)
        try:
            for epoch in range(start_epoch, epochs):
                train = self.train_one_epoch(train_dataset, epoch)
                validation = self.validate(val_dataset)
                summary = validation["summary"]
                error = summary["relative_error"]
                improved = (summary["invalid_scenes"] == 0 and error is not None
                            and error < self.best_error)
                if improved:
                    self.best_error = error
                report = dict(epoch=epoch + 1, global_step=self.global_step,
                              train=train, validation=validation)
                path = self.output_dir / "metrics" / f"epoch_{epoch + 1:04d}.json"
                path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
                metadata = dict(config=self.config, next_epoch=epoch + 1,
                                global_step=self.global_step, best_error=self.best_error)
                save_checkpoint(self.output_dir / "checkpoints" / "latest.pt", self.model, self.optimizer, **metadata)
                save_model(
                    self.model, self.output_dir, model_config, "model.pt",
                    epoch=epoch + 1, global_step=self.global_step,
                )
                if improved:
                    save_checkpoint(self.output_dir / "checkpoints" / "best.pt", self.model, self.optimizer, **metadata)
                    save_model(
                        self.model, self.output_dir, model_config, "model_best.pt",
                        epoch=epoch + 1, global_step=self.global_step,
                    )
                print(f"Epoch {epoch + 1}: train loss={train['summary']['loss']:.6f}, "
                      f"validation relative error={error}", flush=True)
        finally:
            self.model._restore_original_geometric_input_config()
        return report
