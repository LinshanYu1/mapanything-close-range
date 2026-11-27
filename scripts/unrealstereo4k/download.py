"""下载 UnrealStereo4K 原始版；默认只检查大小，--download 才下载。"""
import argparse
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[2]
BASE = 'https://s3.eu-central-1.amazonaws.com/avg-projects/smd_nets'
DEFAULT_ARGS = ['--scenes', '00000']


def project_path(value):
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def scene_id(value):
    if not re.fullmatch(r'0000[0-8]', value):
        raise argparse.ArgumentTypeError('场景必须是 00000–00008')
    return value


def download(url, archive):
    """先写 .part；中断保留，重跑时以 HTTP Range 继续。"""
    archive.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(urllib.request.Request(url, method='HEAD'), timeout=60) as response:
        total = int(response.headers['Content-Length'])
    if archive.exists():
        if archive.stat().st_size != total or not zipfile.is_zipfile(archive):
            raise ValueError(f'已有压缩包大小/格式异常，请检查：{archive}')
        return
    partial = archive.with_suffix('.zip.part')
    offset = partial.stat().st_size if partial.exists() else 0
    if offset > total:
        raise ValueError(f'临时文件比远端更大，请检查：{partial}')
    if offset < total:
        request = urllib.request.Request(url, headers={'Range': f'bytes={offset}-'} if offset else {})
        with urllib.request.urlopen(request, timeout=120) as response:
            if offset and (response.status != 206 or not response.headers.get('Content-Range', '').startswith(f'bytes {offset}-')):
                raise ValueError('服务器未按要求续传；保留本地文件，不追加错误内容')
            with partial.open('ab' if offset else 'wb') as output:
                reported = offset
                while chunk := response.read(8 * 1024 * 1024):
                    output.write(chunk)
                    offset += len(chunk)
                    if offset - reported >= 256 * 1024 * 1024:
                        print(f'{archive.name}: {offset / total:.1%}', flush=True)
                        reported = offset
    if partial.stat().st_size != total or not zipfile.is_zipfile(partial):
        raise ValueError('下载不完整或不是 ZIP；.part 已保留')
    partial.replace(archive)


def extract(archive, raw_root, scene):
    destination = raw_root / scene
    if destination.exists():
        raise FileExistsError(f'不覆盖已有场景：{destination}')
    with tempfile.TemporaryDirectory(prefix='.extract-', dir=raw_root) as temporary:
        staging = Path(temporary)
        with zipfile.ZipFile(archive) as package:
            for member in package.infolist():
                target = (staging / member.filename).resolve()
                if not target.is_relative_to(staging.resolve()) or ((member.external_attr >> 16) & 0o170000) == 0o120000:
                    raise ValueError(f'不安全的 ZIP 路径：{member.filename}')
            package.extractall(staging)  # 解压过程验证 ZIP CRC。
        candidates = [p.parent for p in staging.rglob('Image0') if p.is_dir()]
        if len(candidates) != 1:
            raise ValueError('压缩包内无法唯一定位场景 Image0')
        source = candidates[0]
        for folder in ('Image0', 'Image1', 'Disp0', 'Disp1', 'Extrinsics0', 'Extrinsics1'):
            if not (source / folder).is_dir():
                raise ValueError(f'压缩包缺少 {folder}')
        shutil.move(str(source), str(destination))
    print(f'解压完成：{destination}', flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scenes', nargs='+', type=scene_id, default=['00000'])
    parser.add_argument('--raw-root', default='data/raw/unrealstereo4k')
    parser.add_argument('--download', action='store_true')
    parser.add_argument('--extract', action='store_true')
    args = parser.parse_args(argv)
    raw_root = project_path(args.raw_root)
    for scene in dict.fromkeys(args.scenes):
        if scene == '00008':
            print('注意：00008 是官方域外测试场景，不用于训练。', flush=True)
        url = f'{BASE}/UnrealStereo4K_{scene}.zip'
        archive = raw_root / 'archives' / f'UnrealStereo4K_{scene}.zip'
        if args.download:
            download(url, archive)
        elif not args.extract:
            with urllib.request.urlopen(urllib.request.Request(url, method='HEAD'), timeout=60) as response:
                print(f'{scene}: HTTP {response.status}, {int(response.headers["Content-Length"]) / 1e9:.2f} GB; {url}')
        if args.extract:
            raw_root.mkdir(parents=True, exist_ok=True)
            extract(archive, raw_root, scene)


if __name__ == '__main__':
    main(sys.argv[1:] or DEFAULT_ARGS)
