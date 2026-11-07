from pathlib import Path
import os,sys,shutil,time,json
P=Path(__file__).resolve().parents[1];R=P/'app/flowhmr';os.chdir(R);sys.path.insert(0,str(R))
os.environ.update(FLOWHMR_UNLOAD_BETWEEN_STAGES='1',POSTPROCESS_SKIP_SMOOTH='1',SKIP_KEYPOINT_PROMPT='1',MHR_USE_CUDA_GRAPH='0',LAYER_DTYPE='fp32',DINOV3_LOCAL_REPO=str(R/'third_party/dinov3_hub/facebookresearch_dinov3_main'))
import torch,numpy as np
from flowhmr.pipeline.v2m_inference import V2MInferenceEngine
O=R/'output/example';B=P/'logs/staged_validation_backup';B.mkdir(exist_ok=True)
f=O/'example_sam3d_feat.pt';backup=B/f.name;shutil.copy2(f,backup)
assert f.resolve()==(R/'output/example/example_sam3d_feat.pt').resolve();f.unlink()
E=V2MInferenceEngine.build(str(R/'checkpoints/flowhmr_latest/flowhmr_latest.ckpt'),str(R/'configs/base/model_basic05B.yml'),device='cuda',postprocess=True)
torch.cuda.reset_peak_memory_stats();t=time.perf_counter()
try:
    E.process_video(str(P/'inputs/example.mp4'),str(R/'output'),vggt_frame_interval=10)
    z0=np.load(O/'flowhmr_latest.npz');z1=np.load(O/'example_seed0.npz')
    r={'seconds':time.perf_counter()-t,'peak_alloc_GiB':torch.cuda.max_memory_allocated()/2**30,'peak_reserved_GiB':torch.cuda.max_memory_reserved()/2**30,'max_abs_diff':{k:float(np.max(np.abs(z0[k]-z1[k]))) for k in ('poses','trans','betas')}}
    a=torch.load(backup,map_location='cpu',weights_only=True);b=torch.load(f,map_location='cpu',weights_only=True)
    r['feature_max_abs_diff']=float((a-b).abs().max());r['feature_bit_equal']=torch.equal(a,b)
    (P/'logs/staged_cold_sam_validation.json').write_text(json.dumps(r,indent=2),encoding='utf-8');print('COLD_SAM',json.dumps(r),flush=True)
finally:
    if not f.is_file():shutil.copy2(backup,f)
