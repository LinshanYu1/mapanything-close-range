"""下载全部官方 GraspNet 场景包；续传、CRC 检查和安全解压。"""
import argparse
import concurrent.futures
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import threading
import time
import zipfile

import gdown

PACKS = {
    'train_1':'1wQx8IJ_Lok3hVK_nchQUzgw88QBGZ5iq',
    'train_2':'1b1Z1goPV0o_wdwXZ8qTlHd2TBRU5-CmH',
    'train_3':'1oNcmZno2ymsDUWTmfFOxewMBjTXhL95c',
    'train_4':'1e8Xy7-lFhiXk0ugPOKvHKDiGTparmx00',
    'test_seen':'1_nxiCmHhtsjCgA1IKJn3AuMq4SH_fseW',
    'test_similar':'1njgthC-uUvTXgG99qq1fjS-fzofttFms',
    'test_novel':'1xixvgY0yK7TEALq3k7JcJk2_SP_6r8nk',
}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path(__file__).resolve().parents[2]/'data/raw/graspnet')
    parser.add_argument('--workers',type=int,default=2)
    parser.add_argument('--download-only',action='store_true')
    args=parser.parse_args()
    root=args.output.resolve();archives=root/'archives';archives.mkdir(parents=True,exist_ok=True)
    status={};lock=threading.Lock()
    def update(name,**values):
        with lock:
            status.setdefault(name,{}).update(values)
            temp=root/'download_status.json.tmp'
            temp.write_text(json.dumps(status,indent=2))
            os.replace(temp,root/'download_status.json')
        print(name,values,flush=True)
    def work(name,file_id):
        archive=archives/f'{name}.zip';partial=archives/f'{name}.zip.part'
        try:
            update(name,state='downloading',source=f'https://drive.google.com/file/d/{file_id}/view')
            if not archive.exists():
                result=gdown.download(id=file_id,output=str(partial),quiet=True,resume=True)
                if result is None or not zipfile.is_zipfile(partial):raise RuntimeError('No valid ZIP downloaded')
                partial.replace(archive)
            update(name,state='verifying',bytes=archive.stat().st_size)
            with zipfile.ZipFile(archive) as z:
                if args.download_only:
                    bad=z.testzip()
                    if bad:raise RuntimeError(f'CRC failure: {bad}')
                else:
                    update(name,state='extracting')
                    for member in z.infolist():
                        path=PurePosixPath(member.filename)
                        if path.is_absolute() or '..' in path.parts:raise ValueError('Unsafe archive path')
                        if member.is_dir():continue
                        # Keep archive directory structure. CRC is verified while reading.
                        target=root/Path(*path.parts);target.parent.mkdir(parents=True,exist_ok=True)
                        if target.exists():
                            if target.stat().st_size!=member.file_size:raise ValueError(f'Existing size mismatch: {target}')
                            continue
                        temp=target.with_suffix(target.suffix+'.part')
                        with z.open(member) as src,temp.open('wb') as dst:shutil.copyfileobj(src,dst,1024*1024)
                        temp.replace(target)
            update(name,state='complete')
        except Exception as e:
            update(name,state='failed',error=str(e))
            raise
    failures=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures={pool.submit(work,n,i):n for n,i in PACKS.items()}
        for future in concurrent.futures.as_completed(futures):
            try:future.result()
            except Exception:failures.append(futures[future])
    if failures:raise SystemExit('Failed packs: '+', '.join(failures))
    print('All seven scene archives complete.',flush=True)

if __name__=='__main__':main()
