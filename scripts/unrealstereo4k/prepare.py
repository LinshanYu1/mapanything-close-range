"""原始 UnrealStereo4K → 项目统一格式；不改原图，不覆盖已有输出。

相机参数读取和 baseline * fx / disparity 遵循 upstream conversion/unrealstereo4k.py。
必须确认原始外参长度单位，再填写 --units-to-meters（米=1，厘米=0.01）。
目前仅支持原始 PNG 版，不自动猜测缩小版相机参数。
"""
import argparse
from pathlib import Path
import sys
import tempfile
import shutil

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import minimum_filter, maximum_filter, binary_erosion

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from close_range.common import portable_path, save, sha
from mapanything.utils.cropping import crop_resize_if_necessary
from mapanything.utils.image import find_closest_aspect_ratio

# 直接 Run 时在这里填写参数；长度单位必须先从数据来源确认。
DEFAULT_ARGS = ['--scene', '00000', '--frames', '00000']


def camera(path):
    lines = path.read_text().strip().splitlines()
    if len(lines) != 2:
        raise ValueError(f'相机参数应为两行：{path}')
    intrinsic = np.fromstring(lines[0], sep=' ').reshape(3, 3)
    extrinsic = np.eye(4)
    extrinsic[:3] = np.fromstring(lines[1], sep=' ').reshape(3, 4)
    if not np.isfinite(intrinsic).all() or not np.isfinite(extrinsic).all() or intrinsic[0, 0] <= 0 or intrinsic[1, 1] <= 0:
        raise ValueError(f'无效相机参数：{path}')
    np.linalg.inv(intrinsic)
    np.linalg.inv(extrinsic)
    return intrinsic, extrinsic


def prepare_scene(args, scene):
    raw = (ROOT / args.raw_root / scene).resolve()
    frames = args.frames or [p.stem for p in sorted((raw / 'Image0').glob('*.png'))]
    if not frames:
        raise ValueError(f'没有原始 PNG：{raw}')
    for frame in dict.fromkeys(frames):
        if not frame.isdigit():
            raise ValueError('frame 必须是数字文件名')
        name = f'{scene}_{frame}'
        image_dir = ROOT / 'data/processed/unrealstereo4k' / name
        annotation = ROOT / 'data/annotations/unrealstereo4k' / f'{name}.json'
        manifest = ROOT / 'data/manifests/unrealstereo4k' / f'{name}.json'
        if any(p.exists() for p in (image_dir, annotation, manifest)):
            raise FileExistsError(f'已有输出，不覆盖：{name}')
        parameters = [camera(raw / f'Extrinsics{c}' / f'{frame}.txt') for c in (0, 1)]
        k, e = parameters[0]
        baseline = float(np.linalg.norm((e @ np.linalg.inv(parameters[1][1]))[:3, 3]))
        if not np.isfinite(baseline) or baseline <= 0:
            raise ValueError('相机基线必须为正')
        images = [Image.open(raw / f'Image{c}' / f'{frame}.png').convert('RGB') for c in (0, 1)]
        if images[0].size != images[1].size:
            raise ValueError('左右图尺寸不一致')
        disparity = np.load(raw / 'Disp0' / f'{frame}.npy', allow_pickle=False)
        if disparity.shape != (images[0].height, images[0].width):
            raise ValueError('RGB/视差尺寸不一致；不支持自动缩小版转换')
        depth = np.zeros(disparity.shape, dtype=np.float64)
        valid = np.isfinite(disparity) & (disparity > 0)
        np.divide(baseline * args.units_to_meters * k[0, 0], disparity, out=depth, where=valid)
        valid &= np.isfinite(depth) & (depth > 0) & (depth <= args.max_depth_m)
        # 排除无效值附近及深度突变边界，避免端点落在遮挡轮廓。
        valid = binary_erosion(valid, iterations=2)
        valid &= maximum_filter(depth, size=5) - minimum_filter(depth, size=5) < .05 * depth
        size = find_closest_aspect_ratio(images[0].width / images[0].height, 518)
        ids = np.arange(depth.size, dtype=np.int32).reshape(depth.shape)
        processed, quantities = crop_resize_if_necessary(images[0], resolution=size, additional_quantities=[ids])
        sy, sx = np.divmod(np.asarray(quantities[0], dtype=np.int64), images[0].width)
        candidates = np.argwhere(valid[sy, sx])
        rng = np.random.default_rng(args.seed)
        pairs, used = [], set()
        for _ in range(args.pairs * 1000):
            if len(candidates) < 2 or len(pairs) == args.pairs:
                break
            xy = candidates[rng.choice(len(candidates), 2, replace=False)][:, ::-1]
            key = tuple(sorted(map(tuple, xy.tolist())))
            if key in used:
                continue
            x, y = xy.T
            original = np.c_[sx[y, x], sy[y, x]]
            rays = np.linalg.solve(k, np.c_[original, np.ones(2)].T).T
            xyz = rays * depth[original[:, 1], original[:, 0], None]
            distance = float(np.linalg.norm(xyz[0] - xyz[1]))
            if not np.isfinite(distance) or distance <= 0:
                continue
            used.add(key)
            pairs.append(dict(id=f'pair_{len(pairs)+1:03d}', points_xy=xy.tolist(), source_points_xy=original.tolist(), distance_m=distance, label_source='synthetic_disparity', label_valid=True))
        if len(pairs) != args.pairs:
            raise ValueError(f'{name}: 有效点对不足')
        image_dir.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=image_dir.parent, prefix='.prepare-') as temporary:
            staged = Path(temporary) / name
            staged.mkdir()
            views = []
            for c, image in enumerate(images):
                dest = staged / f'view_{c+1:03d}.png'
                crop_resize_if_necessary(image, resolution=size)[0].save(dest)
                views.append(dict(id=f'view_{c+1:03d}', image=portable_path(image_dir / dest.name), sha256=sha(dest), original=portable_path(raw / f'Image{c}' / f'{frame}.png')))
            shutil.move(str(staged), str(image_dir))
        save(annotation, dict(scene_id=name, reference_view_id=views[0]['id'], image_sha256=views[0]['sha256'], coordinate_space='model_input_pixels_xy', pairs=pairs, units_to_meters=args.units_to_meters, baseline_raw=baseline, note='Synthetic geometry labels; units supplied by operator.'))
        save(manifest, dict(schema_version=1, scene_id=name, source='unrealstereo4k', original_scene=scene, official_split='test' if scene == '00008' else 'train', views=views, reference_index=0, model_size=list(size), annotation=portable_path(annotation), seed=args.seed))
        preview = processed.copy()
        draw = ImageDraw.Draw(preview)
        for label, (x, y) in zip('AB', pairs[0]['points_xy']):
            draw.ellipse((x-4, y-4, x+4, y+4), outline='red', width=2)
            draw.text((x+6, y), label, fill='red')
        preview.save(annotation.with_suffix('.preview.png'))
        print(f'{name}: 2 views, {len(pairs)} pairs → {portable_path(manifest)}', flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-root', default='data/raw/unrealstereo4k')
    parser.add_argument('--scene', choices=[f'{i:05d}' for i in range(9)])
    parser.add_argument('--frames', nargs='+', help='不指定时处理场景内全部左右图对')
    parser.add_argument('--units-to-meters', type=float, required=True)
    parser.add_argument('--max-depth-m', type=float, default=1000)
    parser.add_argument('--pairs', type=int, default=10)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args(argv)
    if args.pairs < 1 or not np.isfinite(args.units_to_meters) or args.units_to_meters <= 0 or not np.isfinite(args.max_depth_m) or args.max_depth_m <= 0:
        parser.error('pairs、units-to-meters、max-depth-m 必须为有限正数')
    root = ROOT / args.raw_root
    scenes = [args.scene] if args.scene else [f'{i:05d}' for i in range(9) if (root / f'{i:05d}').is_dir()]
    if not scenes:
        parser.error('没有发现原始场景')
    for scene in scenes:
        prepare_scene(args, scene)


if __name__ == '__main__':
    main(sys.argv[1:] or DEFAULT_ARGS)
