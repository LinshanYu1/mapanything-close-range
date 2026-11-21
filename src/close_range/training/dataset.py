"""Phone / GraspNet 场景训练数据集。"""

import hashlib
import json
import math
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset
from PIL import Image

from mapanything.utils.image import load_images
from mapanything.utils.inference import (
    validate_input_views_for_inference,
    preprocess_input_views_for_inference,
)


class SceneDistanceDataset(Dataset):
    """
    一条数据对应一个场景。

    每个场景返回：
        默认启用一致性采样：至少三张图时返回两组不同的双视角输入。
        单张、两张，或关闭一致性采样时：全部照片构成第一组，第二组为 None。
        所有组均将参考图放在第一位，不复制照片凑数。
        参考图上的有效点对坐标
        对应的真实距离，单位为米

    manifest_paths 必须来自提前划分好的训练或验证清单。
    本类不负责划分训练集、验证集和测试集。
    """
    def __init__(
        self,
        project_root,
        manifest_paths,
        training=True,
        seed=0,
        use_consistency=True,
    ):
        self.project_root = Path(project_root).resolve()
        self.training = training
        self.seed = seed
        self.use_consistency = use_consistency
        self.epoch = 0

        paths = [self._resolve(path) for path in manifest_paths]

        if not paths:
            raise ValueError("场景清单不能为空")

        if len(paths) != len(set(paths)):
            raise ValueError("场景清单中存在重复的 manifest")

        # 初始化时检查文件和标签，尽早发现数据问题。
        self.scenes = [self._read_scene(path) for path in paths]

    def _resolve(self, path):
        """相对路径统一以项目根目录为基准。"""
        path = Path(path)

        if not path.is_absolute():
            path = self.project_root / path

        return path.resolve()

    @staticmethod
    def _read_json(path):
        with path.open("r", encoding="utf-8") as file:
            return json.load(file)

    @staticmethod
    def _sha256(path):
        digest = hashlib.sha256()

        with path.open("rb") as file:
            for chunk in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(chunk)

        return digest.hexdigest()
    def _read_scene(self, manifest_path):
        """读取场景信息，并校验图片和标注。"""
        manifest = self._read_json(manifest_path)

        scene_id = manifest["scene_id"]
        source = manifest["source"]
        views = manifest["views"]
        reference_index = manifest["reference_index"]

        if source not in {"phone", "graspnet"}:
            raise ValueError(f"{scene_id}: 不支持的数据来源 {source}")

        if len(views) < 1:
            raise ValueError(
                f"{scene_id}: 场景至少需要一张照片"
            )

        if not 0 <= reference_index < len(views):
            raise ValueError(f"{scene_id}: 参考图编号越界")

        view_ids = [view["id"] for view in views]
        if len(view_ids) != len(set(view_ids)):
            raise ValueError(f"{scene_id}: 照片编号重复")

        width, height = manifest["model_size"]

        if (
                width <= 0
                or height <= 0
                or width % 14 != 0
                or height % 14 != 0
        ):
            raise ValueError(f"{scene_id}: 图片尺寸必须为正的 14 倍数")

        image_paths = []

        for view in views:
            image_path = self._resolve(view["image"])

            if self._sha256(image_path) != view["sha256"]:
                raise ValueError(f"{scene_id}: 图片已变化：{image_path}")

            with Image.open(image_path) as image:
                if image.size != (width, height):
                    raise ValueError(
                        f"{scene_id}: 图片尺寸与 manifest 不一致"
                    )

            image_paths.append(image_path)

        if len(image_paths) != len(set(image_paths)):
            raise ValueError(f"{scene_id}: 多个视角指向同一图片文件")

        annotation = self._read_json(
            self._resolve(manifest["annotation"])
        )
        reference = views[reference_index]

        if annotation["scene_id"] != scene_id:
            raise ValueError(f"{scene_id}: 标注场景不匹配")

        if annotation["reference_view_id"] != reference["id"]:
            raise ValueError(f"{scene_id}: 标注参考图不匹配")

        if annotation["image_sha256"] != reference["sha256"]:
            raise ValueError(f"{scene_id}: 标注对应的图片版本不匹配")

        if annotation["coordinate_space"] != "model_input_pixels_xy":
            raise ValueError(f"{scene_id}: 标注必须使用模型输入图片坐标")

        pairs = []

        for pair in annotation["pairs"]:
            # 已明确标记无效的标签不参与训练。
            if not pair.get("label_valid", False):
                continue

            points = torch.tensor(
                pair["points_xy"],
                dtype=torch.float32,
            )
            distance = float(pair["distance_m"])

            if points.shape != (2, 2):
                raise ValueError(f"{scene_id}: 每个点对必须包含两个 xy 坐标")

            if not torch.isfinite(points).all():
                raise ValueError(f"{scene_id}: 点坐标包含无效数值")

            if (
                    (points < 0).any()
                    or (points[:, 0] > width - 1).any()
                    or (points[:, 1] > height - 1).any()
            ):
                raise ValueError(f"{scene_id}: 标注点越界")

            # 与现有脚本保持一致，后续按最近像素读取三维点。
            if torch.equal(points[0].round(), points[1].round()):
                raise ValueError(f"{scene_id}: 两个端点落在同一个像素")

            if not math.isfinite(distance) or distance <= 0:
                raise ValueError(f"{scene_id}: 真实距离必须为有限正数")

            pairs.append(
                {
                    "id": pair["id"],
                    "points_xy": points,
                    "distance_m": distance,
                }
            )

        if not pairs:
            raise ValueError(f"{scene_id}: 没有可用于训练的有效点对")

        return {
            "scene_id": scene_id,
            "source": source,
            "views": views,
            "image_paths": image_paths,
            "reference_index": reference_index,
            "model_size": (width, height),
            "pairs": pairs,
        }

    def __len__(self):
        return len(self.scenes)

    def set_epoch(self, epoch):
        """训练器每轮开始时调用，让训练视角组合随轮次变化。"""
        self.epoch = epoch

    def _select_supports(self, scene):
        """选择两张不同的辅助照片，参考图始终不变。"""
        candidates = [
            index
            for index in range(len(scene["views"]))
            if index != scene["reference_index"]
        ]

        if len(candidates) < 2:
            raise ValueError("一致性采样需要至少两张辅助照片")

        # 验证集每轮保持同一组合。
        epoch = self.epoch if self.training else 0

        # 使用稳定的场景级种子，不依赖 Python 的随机 hash。
        seed_text = (
            f"{self.seed}:{epoch}:"
            f"{scene['source']}:{scene['scene_id']}"
        )
        digest = hashlib.sha256(seed_text.encode("utf-8")).digest()
        local_seed = int.from_bytes(digest[:8], byteorder="big")

        rng = random.Random(local_seed)
        return rng.sample(candidates, k=2)

    def _load_subset(self, scene, indices):
        """按给定编号加载一组图片，参考图必须在第一位。"""
        paths = [str(scene["image_paths"][index]) for index in indices]

        views = load_images(
            paths,
            resize_mode="fixed_size",
            size=scene["model_size"],
        )

        if len(views) != len(indices):
            raise ValueError(f"{scene['scene_id']}: 图片加载数量不正确")

        width, height = scene["model_size"]

        for view in views:
            if tuple(view["img"].shape[-2:]) != (height, width):
                raise ValueError(
                    f"{scene['scene_id']}: 加载过程改变了标注对应的图片尺寸"
                )

        views = validate_input_views_for_inference(views)
        views = preprocess_input_views_for_inference(views)

        view_ids = [scene["views"][index]["id"] for index in indices]
        return views, view_ids

    def __getitem__(self, index):
        scene = self.scenes[index]

        reference = scene["reference_index"]
        views_2 = None
        view_ids_2 = None

        if self.use_consistency and len(scene["views"]) >= 3:
            support_1, support_2 = self._select_supports(scene)
            indices_1 = [reference, support_1]
            views_2, view_ids_2 = self._load_subset(
                scene, [reference, support_2]
            )
        else:
            # 普通单组输入：一张也可以，不丢弃其他已有视角。
            indices_1 = [reference] + [
                i for i in range(len(scene["views"])) if i != reference
            ]

        views_1, view_ids_1 = self._load_subset(scene, indices_1)

        return {
            "scene_id": scene["scene_id"],
            "source": scene["source"],
            "views_1": views_1,
            "views_2": views_2,
            "view_ids_1": view_ids_1,
            "view_ids_2": view_ids_2,
            "pair_ids": [pair["id"] for pair in scene["pairs"]],
            "points_xy": torch.stack(
                [pair["points_xy"] for pair in scene["pairs"]]
            ),
            "target_distances_m": torch.tensor(
                [pair["distance_m"] for pair in scene["pairs"]],
                dtype=torch.float32,
            ),
        }
