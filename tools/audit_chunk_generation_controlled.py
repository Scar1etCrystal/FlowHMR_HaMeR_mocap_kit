"""Controlled generation-only audit: exact feature/camera/global-noise slices.
No detection, VGGT, SAM or postprocess rerun; no production-source edits.
Local control, not the old cloud control. Review G files are raw, no postprocess.
"""
from pathlib import Path
import sys,os,json,time
import torch,numpy as np
from scipy.spatial.transform import Rotation as R,Slerp
P=Path(__file__).resolve().parents[1];REPO=P/'app/flowhmr';sys.path.insert(0,str(REPO));os.chdir(REPO)
from flowhmr.pipeline.v2m_inference import V2MInferenceEngine,build_feature_dict,_rotate_smplh_params_y180
import flowhmr.pipeline.pipeline_v2m as pm
from flowhmr.utils.runtime_vggt_camera import _vggt_opencv_to_yup_rt
from flowhmr.core.math.geometry import rot6d_to_rotation_matrix,rotation_matrix_to_angle_axis
from flowhmr.core.export.smplh2fbx import SMPLH2FBX
D=P/'outputs/manual_review';D.mkdir(exist_ok=True)
O=REPO/'output/chunk_full';tok=torch.load(O/'chunk_full_sam3d_feat.pt',map_location='cpu',weights_only=True)
c=np.load(O/'chunk_full_vggt_camera.npz');cam=_vggt_opencv_to_yup_rt(torch.from_numpy(c['RT_vggt'].astype(np.float32)))
noise=pm.randn_tensor((1,300,491),generator=torch.Generator().manual_seed(0),dtype=torch.float32)
E=V2MInferenceEngine.build_lite(str(REPO/'checkpoints/flowhmr_latest/flowhmr_latest.ckpt'),str(REPO/'configs/base/model_basic05B.yml'),device='cuda',postprocess=False)
result={};original=pm.randn_tensor
for tag,start,n in [('full',0,300),('truncA',0,150),('truncB',150,150),('overA',0,195),('overB',105,195)]:
    called=[]
    def fixed(shape,generator=None,device=None,dtype=None,layout=None):
        assert tuple(shape)==(1,n,491),(shape,n)
        v=noise[:,start:start+n].clone().to(device=device or 'cpu',dtype=dtype or torch.float32);called.append(True);return v
    pm.randn_tensor=fixed
    try:
        feature=build_feature_dict(tok[start:start+n],cam[start:start+n],E.device)
        torch.cuda.reset_peak_memory_stats();t=time.perf_counter()
        with torch.no_grad():out=E.pipeline.generate(feature=feature,seeds=[0],length=n,cfg_scale=1.0)
        assert called==[True],called
        p=rotation_matrix_to_angle_axis(rot6d_to_rotation_matrix(out['rot6d'][0])).cpu().numpy()
        tr=out['trans'][0].cpu().numpy().copy();_rotate_smplh_params_y180(p,tr)
        result[tag]={'poses':p,'trans':tr,'betas':out['shapes'][0,:1].cpu().numpy(),'seconds':time.perf_counter()-t,'peak_alloc_GiB':torch.cuda.max_memory_allocated()/2**30}
        del out,feature;torch.cuda.empty_cache();print('GENERATED',tag,start,n,flush=True)
    finally:pm.randn_tensor=original

def geo(a,b):return np.degrees((R.from_rotvec(a.reshape(-1,3)).inv()*R.from_rotvec(b.reshape(-1,3))).magnitude()).reshape(-1,52)
def stats(v):return {'mean':float(v.mean()),'p99':float(np.percentile(v,99)),'max':float(v.max())}
f=result['full'];audit={}
for tag,start,n in [('truncA',0,150),('truncB',150,150),('overA',0,195),('overB',105,195)]:
    d=geo(f['poses'][start:start+n],result[tag]['poses'])
    audit[tag]={'body_local_ex_root_deg':stats(d[:,1:22]),'fingers_deg':stats(d[:,22:]),'input_features_equal_to_global_slice':True,'input_camera_equal_to_global_slice':True,'initial_noise_equal_to_global_slice':True}
ov=geo(result['overA']['poses'][105:195],result['overB']['poses'][:90]);audit['overlap_disagreement']={'body_local_ex_root_deg':stats(ov[:,1:22]),'fingers_deg':stats(ov[:,22:])}
# Same aligned hard-cut and crossfade comparison as review C group, raw generation only.
A=result['overA'];B=result['overB'];rel=R.from_rotvec(A['poses'][105:195,0])*R.from_rotvec(B['poses'][:90,0]).inv();yaw=float(np.median(rel.as_euler('yxz')[:,0]));ry=R.from_euler('y',yaw)
bp=B['poses'].copy();bp[:,0]=(ry*R.from_rotvec(bp[:,0])).as_rotvec();bt=ry.apply(B['trans']-B['trans'][:90].mean(0))+A['trans'][105:195].mean(0)
hp=np.concatenate([A['poses'][:149],bp[44:]]);ht=np.concatenate([A['trans'][:149],bt[44:]])
w=np.linspace(0,1,90);mid=A['poses'][105:195].copy()
for j in range(52):
    for i in range(90):mid[i,j]=Slerp([0,1],R.from_rotvec(np.stack([A['poses'][105+i,j],bp[i,j]])))([w[i]]).as_rotvec()[0]
bp2=np.concatenate([A['poses'][:105],mid,bp[90:]]);bt2=np.concatenate([A['trans'][:105],(1-w[:,None])*A['trans'][105:195]+w[:,None]*bt[:90],bt[90:]])
cases={'G00_controlled_whole_raw':(f['poses'],f['trans']),'G01_controlled_trunc_raw':(np.concatenate([result['truncA']['poses'],result['truncB']['poses']]),np.concatenate([result['truncA']['trans'],result['truncB']['trans']])),'G02_controlled_overlap_hardcut_raw':(hp,ht),'G03_controlled_overlap_crossfade_raw':(bp2,bt2)}
manifest=[]
for name,(p,t) in cases.items():
    z={'poses':p.astype(np.float32),'trans':t.astype(np.float32),'betas':f['betas'],'Rh':p[:,0].astype(np.float32),'mocap_framerate':30,'num_frames':len(p),'gender':'neutral'}
    npz=D/f'{name}.npz';fbx=D/f'{name}.fbx';np.savez(npz,**z)
    conv=SMPLH2FBX(str(REPO/'assets/body_models/smplh/neutral/model.npz'),obj_template_path=None);assert conv.convert_npz_to_fbx(str(npz),str(fbx));conv.mgr.Destroy()
    manifest.append({'name':name,'frames':len(p),'fbx':str(fbx),'npz':str(npz),'bytes':fbx.stat().st_size})
    audit[name]={'body_local_ex_root_deg':stats(geo(f['poses'],p)[:,1:22])}
r={'conditions':'same original FP32 weights,20steps,cfg1; global cached tokens and camera; global CPU noise sliced at matching frame indices; NO postprocess; NO HaMeR; local control','audit':audit,'exports':manifest,'raw_chunk_beta_L2_difference':{tag:float(np.linalg.norm(v['betas']-f['betas'])) for tag,v in result.items()},'assembled_exports_use_common_full_betas':True}
(D/'controlled_generation_audit.json').write_text(json.dumps(r,indent=2),encoding='utf-8');print('CONTROLLED',json.dumps(r),flush=True)
