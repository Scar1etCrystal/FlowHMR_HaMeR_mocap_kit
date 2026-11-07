"""Fetch exact long-run caches. Password comes only from the process environment."""
from pathlib import Path
import sys,os,json,hashlib,shlex,time
P=Path(__file__).resolve().parents[1];sys.path.insert(0,str(P/'tools/transfer_deps'))
import paramiko
D=P/'inputs/violent_original_cache';D.mkdir(parents=True,exist_ok=True)
root='/root/autodl-fs/mocap_fsb_2026-10-10/FlowHMR';out=root+'/output/video_cut_1080p30'
c=paramiko.SSHClient();c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect('connect.westb.seetacloud.com',port=27820,username='root',password=os.environ['FLOW_CACHE_SSH_PASSWORD'],look_for_keys=False,allow_agent=False,timeout=15,auth_timeout=20)
s=c.open_sftp();listing=s.listdir(out);print('REMOTE_CACHE_FILES',listing,flush=True)
manifest={'host':'connect.westb.seetacloud.com','port':27820,'remote_directory':out,'files':[],'no_model_inference_on_cloud':True}
def exists(p):
    try:s.stat(p);return True
    except FileNotFoundError:return False
    except OSError:return False
def fetch(src,name=None):
    dest=D/(name or Path(src).name);size=s.stat(src).st_size
    print('FETCH',src,size,flush=True);s.get(src,str(dest))
    h=hashlib.sha256(dest.read_bytes()).hexdigest();_,stdout,stderr=c.exec_command('sha256sum -- '+shlex.quote(src),timeout=60)
    remote=stdout.read().decode().split()[0];assert h==remote,(src,h,remote);assert dest.stat().st_size==size
    manifest['files'].append({'remote':src,'local':str(dest),'bytes':size,'sha256':h});print('VERIFIED',dest.name,size,flush=True)
for n in ('video_cut_1080p30_sam3d_feat.pt','video_cut_1080p30_vggt_camera.npz','video_cut_1080p30_bbox.npz'):
    assert n in listing,('Missing original cache',n);fetch(out+'/'+n)
for src,name in [(root+'/checkpoints/flowhmr_latest/config.yml','original_checkpoint_config.yml'),(root+'/configs/base/model_basic05B.yml','original_base_config.yml')]:
    if exists(src):fetch(src,name)
# Reconstruct the original seed on its own Linux PyTorch CPU, never load a model.
py='/root/autodl-tmp/envs/flowhmr/bin/python';noise='/root/autodl-tmp/violent_overlap_global_noise_seed0.pt'
if exists(py):
    code="import torch; z=torch.load("+repr(out+'/video_cut_1080p30_sam3d_feat.pt')+",map_location='cpu',weights_only=True); n=z.shape[0]; g=torch.Generator().manual_seed(0); a=torch.randn((1,n,491),generator=g,dtype=torch.float32);torch.save(a,"+repr(noise)+");print('NOISE',n,torch.__version__)"
    command='OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 '+shlex.quote(py)+' -c '+shlex.quote(code)
    _,stdout,stderr=c.exec_command(command,timeout=120);text=stdout.read().decode();err=stderr.read().decode();rc=stdout.channel.recv_exit_status();print(text.strip(),flush=True)
    if rc==0:fetch(noise,'global_noise_seed0.pt');manifest['noise']='Reconstructed on original Linux PyTorch CPU at original full length, seed0'
    else:manifest['noise_error']=err[-2000:];print('NOISE_ERROR',manifest['noise_error'],flush=True)
(D/'transfer_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
s.close();c.close();print('TRANSFER_DONE',D,flush=True)
