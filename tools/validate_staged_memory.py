from pathlib import Path
import os,sys,shutil,json,time,gc
P=Path(__file__).resolve().parents[1];R=P/'app/flowhmr';os.chdir(R);sys.path.insert(0,str(R))
os.environ['FLOWHMR_UNLOAD_BETWEEN_STAGES']='1';os.environ['POSTPROCESS_SKIP_SMOOTH']='1'
os.environ['DINOV3_LOCAL_REPO']=str(R/'third_party/dinov3_hub/facebookresearch_dinov3_main')
import torch,numpy as np
from flowhmr.pipeline.v2m_inference import V2MInferenceEngine
O=R/'output/chunk_full';B=P/'logs/staged_validation_backup';B.mkdir(exist_ok=True)
cam=O/'chunk_full_vggt_camera.npz';backup=B/cam.name
shutil.copy2(cam,backup)
assert cam.resolve()==(R/'output/chunk_full/chunk_full_vggt_camera.npz').resolve()
cam.unlink()  # Force the camera model to load for residency validation.
engine=V2MInferenceEngine.build(str(R/'checkpoints/flowhmr_latest/flowhmr_latest.ckpt'),str(R/'configs/base/model_basic05B.yml'),device='cuda',postprocess=True)
torch.cuda.reset_peak_memory_stats();t=time.perf_counter()
try:
    engine.process_video(str(P/'inputs/chunktest/chunk_full.mp4'),str(R/'output'),vggt_frame_interval=10)
    result={'seconds':time.perf_counter()-t,'peak_alloc_GiB':torch.cuda.max_memory_allocated()/2**30,'peak_reserved_GiB':torch.cuda.max_memory_reserved()/2**30}
    before=np.load(O/'flowhmr_latest.npz');after=np.load(O/'chunk_full_seed0.npz')
    result['max_abs_diff']={k:float(np.max(np.abs(before[k]-after[k]))) for k in ('poses','trans','betas')}
    a=np.load(backup);b=np.load(cam)
    result['camera_max_abs_diff']={k:float(np.max(np.abs(a[k]-b[k]))) for k in ('K','RT_vggt')}
    result['fbx_bytes']=(O/'chunk_full_seed0.fbx').stat().st_size
    (P/'logs/staged_validation.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print('STAGED_VALIDATION',json.dumps(result),flush=True)
finally:
    if not cam.is_file():shutil.copy2(backup,cam)
