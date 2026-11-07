"""Measure the shipped pipeline's generation memory at LONG lengths.
Repeated real cached tokens/cameras test allocation only, NOT motion quality.
No precision, attention-backend, ODE-count or production-source changes.
Each result is flushed immediately. Do not infer linear scaling from short runs.
"""
from pathlib import Path
import os, sys, time, gc, json, math
P=Path(__file__).resolve().parents[1]; R=P/'app/flowhmr'
os.chdir(R); sys.path.insert(0,str(R))
os.environ['POSTPROCESS_SKIP_SMOOTH']='1'
import torch, numpy as np
from flowhmr.pipeline.v2m_inference import V2MInferenceEngine, build_feature_dict
from flowhmr.utils.runtime_vggt_camera import _vggt_opencv_to_yup_rt
OUT=P/'logs/memory_long_audit.json'
rows=[]
report={'purpose':'memory-only; tiled inputs are not quality evidence', 'torch':torch.__version__, 'gpu':torch.cuda.get_device_name(0),'rows':rows}
def save(): OUT.write_text(json.dumps(report,indent=2),encoding='utf-8')
def snap():
    return {'allocated_GiB':torch.cuda.memory_allocated()/2**30,'reserved_GiB':torch.cuda.memory_reserved()/2**30,'free_driver_GiB':torch.cuda.mem_get_info()[0]/2**30}
def measure(name,T,fn):
    gc.collect();torch.cuda.empty_cache();torch.cuda.synchronize();before=snap();torch.cuda.reset_peak_memory_stats();t=time.perf_counter()
    row={'stage':name,'frames':T,'before':before}
    try:
        result=fn();torch.cuda.synchronize()
        row.update(ok=True,seconds=time.perf_counter()-t,peak_alloc_GiB=torch.cuda.max_memory_allocated()/2**30,peak_reserved_GiB=torch.cuda.max_memory_reserved()/2**30,after=snap())
    except torch.cuda.OutOfMemoryError as e:
        row.update(ok=False,error=str(e));result=None
    rows.append(row);save();print(json.dumps(row),flush=True)
    return result,row
engine=V2MInferenceEngine.build_lite(str(R/'checkpoints/flowhmr_latest/flowhmr_latest.ckpt'),str(R/'configs/base/model_basic05B.yml'),device='cuda',postprocess=True)
report['baseline']=snap();report['steps']=engine.pipeline.validation_steps
report['network_parameters']=sum(p.numel() for p in engine.pipeline.motion_transformer.parameters())
report['dtype']=str(next(engine.pipeline.motion_transformer.parameters()).dtype)
save()
tok=torch.load(R/'output/chunk_full/chunk_full_sam3d_feat.pt',map_location='cpu',weights_only=True)
z=np.load(R/'output/chunk_full/chunk_full_vggt_camera.npz');cam=_vggt_opencv_to_yup_rt(torch.from_numpy(z['RT_vggt'].astype(np.float32)))
# Expose actual temporal dimensions reaching the network, once per length.
seen=set()
def hook(mod,args,kw):
    x=kw.get('x');c=kw.get('ctxt_input')
    T=x.shape[1]
    if T not in seen:
        seen.add(T);print('NETWORK',json.dumps({'motion':list(x.shape),'condition':{k:list(v.shape) for k,v in c.items()},'dtype':str(x.dtype)}),flush=True)
h=engine.pipeline.motion_transformer.register_forward_pre_hook(hook,with_kwargs=True)
for T in [300,900,1500,2100,2700,3000,3300,3794]:
    # Fit baseline + linear + quadratic from completed long GENERATION tests.
    prior=[r for r in rows if r['stage']=='generation' and r.get('ok')]
    if len(prior)>=3:
        xs=np.array([r['frames'] for r in prior],float);ys=np.array([r['peak_alloc_GiB'] for r in prior])
        coef=np.polyfit(xs,ys,2);pred=float(np.polyval(coef,T));report['quadratic_generation_fit']=coef.tolist();save()
        if pred>9.0:
            report['next_length_skipped']={'frames':T,'predicted_alloc_GiB':pred,'reason':'keep desktop and allocator headroom on physical 12GiB'};save();print('STOP_HEADROOM',report['next_length_skipped'],flush=True);break
    feat=build_feature_dict(tok.repeat((math.ceil(T/len(tok)),1))[:T],cam.repeat((math.ceil(T/len(cam)),1,1))[:T],engine.device)
    output,g=measure('generation',T,lambda:engine.pipeline.generate(feature=feat,seeds=[0],length=T,cfg_scale=1.0))
    del feat
    if output is None:break
    # Real postprocess; same skip-smooth flag as production. The generated synthetic
    # motion can have unusual contacts; record that this is NOT a quality test.
    output,pp=measure('postprocess',T,lambda:engine.postprocess.run(output))
    del output;gc.collect();torch.cuda.empty_cache()
    if not pp.get('ok') or max(g.get('peak_alloc_GiB',0),pp.get('peak_alloc_GiB',0))>9.0:
        report['stopped_after_frames']=T;save();break
h.remove();report['finished']=True;save()
