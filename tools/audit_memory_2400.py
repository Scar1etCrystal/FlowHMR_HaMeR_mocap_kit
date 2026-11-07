from pathlib import Path
import os,sys,time,gc,json,math
P=Path(__file__).resolve().parents[1];R=P/'app/flowhmr';os.chdir(R);sys.path.insert(0,str(R))
import torch,numpy as np
from flowhmr.pipeline.v2m_inference import V2MInferenceEngine,build_feature_dict
E=V2MInferenceEngine.build_lite(str(R/'checkpoints/flowhmr_latest/flowhmr_latest.ckpt'),str(R/'configs/base/model_basic05B.yml'),device='cuda',postprocess=False)
tok=torch.load(R/'output/chunk_full/chunk_full_sam3d_feat.pt',map_location='cpu',weights_only=True)
T=2400;feat=build_feature_dict(tok.repeat((math.ceil(T/len(tok)),1))[:T],torch.eye(4).repeat(T,1,1),E.device)
torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();t=time.perf_counter()
out=E.pipeline.generate(feature=feat,seeds=[0],length=T,cfg_scale=1.0);torch.cuda.synchronize()
r={'frames':T,'seconds':time.perf_counter()-t,'peak_alloc_GiB':torch.cuda.max_memory_allocated()/2**30,'peak_reserved_GiB':torch.cuda.max_memory_reserved()/2**30,'free_driver_after_GiB':torch.cuda.mem_get_info()[0]/2**30,'purpose':'tiled cached tokens; memory only, identity camera; 20 steps FP32 cfg1'}
(P/'logs/memory_2400.json').write_text(json.dumps(r,indent=2),encoding='utf-8');print(r)
