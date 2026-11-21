"""完整场景镜像下载：固定版本、SHA-256 校验、安全解压与完整性清单。"""
import argparse,hashlib,json,os,re,tarfile,time
from pathlib import Path,PurePosixPath
from huggingface_hub import HfApi,hf_hub_download


def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--output',type=Path,default=Path(__file__).resolve().parents[2]/'data/raw/graspnet')
 a=p.parse_args();root=a.output.resolve();root.mkdir(parents=True,exist_ok=True)
 repo='DravenALG/GraspNet-1Billion';api=HfApi()
 info=api.dataset_info(repo,files_metadata=True)
 entry=next(f for f in info.siblings if f.rfilename=='scenes.tar')
 status=dict(repo=repo,revision=info.sha,archive_bytes=entry.size,expected_sha256=entry.lfs.sha256,
             source='third-party mirror',official_checksum_verified=False,state='downloading')
 def report(**kwargs):
  status.update(kwargs);temp=root/'mirror_status.json.tmp';temp.write_text(json.dumps(status,indent=2));os.replace(temp,root/'mirror_status.json');print(kwargs,flush=True)
 report(state='downloading')
 archive=Path(hf_hub_download(repo_id=repo,repo_type='dataset',revision=info.sha,filename='scenes.tar',local_dir=root/'archives/mirror'))
 report(state='hashing',archive=os.path.relpath(archive, Path(__file__).resolve().parents[2]))
 digest=hashlib.sha256();last=time.monotonic();read=0
 with archive.open('rb') as f:
  while chunk:=f.read(16*1024*1024):
   digest.update(chunk);read+=len(chunk)
   if time.monotonic()-last>25:report(hashed_bytes=read);last=time.monotonic()
 if digest.hexdigest()!=entry.lfs.sha256:raise RuntimeError('Mirror archive SHA-256 mismatch')
 report(state='extracting',sha256_verified=True)
 count=0;scene_names=set();last=time.monotonic()
 with tarfile.open(archive,mode='r|') as tar:
  for member in tar:
   if member.isdir():continue
   parts=PurePosixPath(member.name).parts
   if member.name.startswith('/') or '..' in parts or not member.isfile():raise ValueError('Unsafe TAR entry')
   indices=[i for i,x in enumerate(parts) if re.fullmatch(r'scene_\d{4}',x)]
   if not indices:continue
   tail=parts[indices[0]:];scene_names.add(tail[0]);target=root/'scenes'/Path(*tail)
   target.parent.mkdir(parents=True,exist_ok=True)
   if target.exists():
    if target.stat().st_size!=member.size:raise ValueError(f'Existing size mismatch: {target}')
   else:
    temp=target.with_suffix(target.suffix+'.part')
    with tar.extractfile(member) as src,temp.open('wb') as dst:
     while block:=src.read(1024*1024):dst.write(block)
    temp.replace(target)
   count+=1
   if time.monotonic()-last>25:report(extracted_files=count,scenes_seen=len(scene_names));last=time.monotonic()
 expected={f'scene_{i:04d}' for i in range(190)}
 if scene_names!=expected:raise RuntimeError(f'Scene mismatch: missing={sorted(expected-scene_names)}')
 inventory=[]
 for scene in sorted(expected):
  for camera in ('realsense','kinect'):
   base=root/'scenes'/scene/camera
   for folder,extension in [('rgb','.png'),('depth','.png'),('label','.png'),('meta','.mat')]:
    missing=[f'{i:04d}{extension}' for i in range(256) if not (base/folder/f'{i:04d}{extension}').is_file()]
    if missing:raise RuntimeError(f'Missing {scene}/{camera}/{folder}: {missing[:5]}')
   inventory.append(dict(scene=scene,camera=camera,rgb=256,depth=256,label=256,meta=256,split='train' if int(scene[6:])<100 else 'test'))
 (root/'scene_inventory.json').write_text(json.dumps(inventory,indent=2))
 report(state='complete',extracted_files=count,scenes=190,rgb_images=190*2*256)

if __name__=='__main__':main()
