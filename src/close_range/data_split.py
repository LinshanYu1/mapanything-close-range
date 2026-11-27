"""共享的数据划分：多来源目录输入，单一训练/验证/测试清单输出。"""
import json
import random
import re
import tempfile
from pathlib import Path

from close_range.common import ROOT, resolve, portable_path

def scene_group(manifest):
    """返回来源和独立分组；目录名不参与数据来源判断。"""
    source = manifest["source"]
    if not isinstance(source, str) or not source.strip():
        raise ValueError("manifest source 必须是非空字符串")
    explicit = manifest.get("split_group")
    if explicit is not None:
        if not isinstance(explicit, str) or not explicit.strip():
            raise ValueError("split_group 必须是非空字符串")
        return source, explicit
    original = manifest.get("original_scene") or manifest["scene_id"]
    if not isinstance(original, str) or not original.strip():
        raise ValueError("原始场景标识必须是非空字符串")
    trajectory = manifest.get("trajectory")
    if trajectory is not None:
        if not isinstance(trajectory, str) or not trajectory.strip():
            raise ValueError("trajectory 必须是非空字符串")
        original = json.dumps([original, trajectory], ensure_ascii=False)
    return source, original


def load_split(path):
    """固定清单；按原始场景和图片指纹检查各集合之间的泄漏。"""
    path = resolve(path).resolve()
    if not path.is_file():
        raise FileNotFoundError("缺少数据划分。预处理完成后先运行：python scripts/tools/make_split.py；此步骤只需执行一次。")
    split = json.loads(path.read_text())
    if not isinstance(split, dict):
        raise ValueError("split 必须是 JSON 对象")
    result, owners, image_owners = {}, {}, {}
    for name in ("train", "val", "test"):
        entries = split.get(name, [])
        if not isinstance(entries, list) or (name != "test" and not entries):
            raise ValueError(f"{name} 必须是非空的 manifest 路径列表")
        paths = []
        for entry in entries:
            if not isinstance(entry, str):
                raise ValueError(f"{name} 中的路径必须是字符串")
            manifest_path = resolve(entry).resolve()
            if manifest_path in paths:
                raise ValueError(f"{name} 存在重复清单：{entry}")
            manifest = json.loads(manifest_path.read_text())
            identity = scene_group(manifest)
            if identity in owners and owners[identity] != name:
                raise ValueError(f"原始场景跨集合重复：{identity}")
            owners[identity] = name
            for view in manifest["views"]:
                digest = view["sha256"]
                if digest in image_owners and image_owners[digest] != name:
                    raise ValueError(f"图片跨集合重复：{entry}")
                image_owners[digest] = name
            paths.append(manifest_path)
        result[name] = paths
    return result


def make_split(manifests_path, output_path, *, seed=443):
    """输入 manifest 目录数组；保留 GraspNet 官方测试集，其他来源按独立分组约 70/15/15。

    固定保留开发场景：GraspNet 0000/0017、Phone scene-001 到验证集。
    分组优先使用 split_group，其次 original_scene + trajectory，最后 scene_id。
    来源动态读取 source；同一来源可跨目录，目录名无需与 source 相同。
    缺目录、缺图片/标签、跨集合重复图片或已有输出都会报错，不静默跳过。
    """
    if isinstance(manifests_path, (str, Path)) or not manifests_path:
        raise ValueError("manifests_path 必须是非空的目录数组")
    output = resolve(output_path)
    if output.exists():
        raise FileExistsError(f"划分已存在，不覆盖：{output}")
    directories = [resolve(p).resolve() for p in manifests_path]
    if len(set(directories)) != len(directories):
        raise ValueError("manifest 目录重复")
    paths, groups = [], {}
    for directory in directories:
        if not directory.is_dir():
            raise FileNotFoundError(f"manifest 目录不存在：{directory}")
        files = sorted(directory.glob("*.json"))
        if not files:
            raise ValueError(f"manifest 目录为空：{directory}")
        for path in files:
            if path in paths:
                raise ValueError(f"清单重复：{path}")
            m = json.loads(path.read_text())
            source, original = scene_group(m)
            if source == "graspnet":
                if not re.fullmatch(r"scene_\d{4}", original) or int(original[6:]) >= 190:
                    raise ValueError(f"GraspNet 官方划分需要 scene_0000 至 scene_0189：{path}")
            groups.setdefault(source, {}).setdefault(original, []).append(portable_path(path))
            paths.append(path)
    # 与训练共用数据校验，不读取测试集预测，也不加载模型权重。
    from close_range.training.dataset import SceneDistanceDataset
    SceneDistanceDataset(ROOT, paths, training=False, seed=seed)
    result = {key: [] for key in ("train", "val", "test")}
    counts = {}
    for source in sorted(groups):
        scenes = groups[source]
        if not scenes:
            continue
        ids = sorted(scenes)
        rng = random.Random(f"{seed}:{source}")
        if source == "graspnet":
            pool = [s for s in ids if int(s[6:]) < 100]
            test = [s for s in ids if int(s[6:]) >= 100]
            val = [s for s in pool if s in {"scene_0000", "scene_0017"}]
            others = [s for s in pool if s not in val]
            # 保留之前完整 GraspNet 划分的随机序列。
            random.Random(seed).shuffle(others)
            needed = max(1, round(len(pool) * .2), len(val)) - len(val)
            val += others[:needed]
            train = others[needed:]
        else:
            if len(ids) < 3:
                raise ValueError(f"{source} 至少需要 3 个独立分组才能划分 train/val/test")
            val = [s for s in ids if source == "phone" and s == "scene-001"]
            others = [s for s in ids if s not in val]
            rng.shuffle(others)
            count = max(1, round(len(ids) * .15))
            needed = max(count, len(val)) - len(val)
            val += others[:needed]
            test = others[needed:needed + count]
            train = others[needed + count:]
        if not train or not val:
            raise ValueError(f"{source} 场景不足以生成非空训练和验证集")
        counts[source] = {name: len(keys) for name, keys in
                          (("train", train), ("val", val), ("test", test))}
        for name, keys in (("train", train), ("val", val), ("test", test)):
            result[name].extend(p for key in sorted(keys) for p in scenes[key])
    result["metadata"] = dict(seed=seed, manifests_path=[portable_path(d) for d in directories],
                              scene_counts=counts,
                              policy="GraspNet official test held out; other sources grouped 70/15/15; development scenes in val")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", dir=output.parent, delete=False) as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
        temporary = Path(f.name)
    try:
        load_split(temporary)
        # 独占创建，防止覆盖另一个进程刚生成的划分。
        with output.open("x") as f:
            f.write(temporary.read_text() + "\n")
    finally:
        temporary.unlink(missing_ok=True)
    print(f"划分已保存：{portable_path(output)}；场景数量：{counts}")
    return result
