"""Violent-motion test: ONLY two overlapping windows + SLERP crossfade.
Original global token/camera/noise slices; existing whole cloud result is control.
Same default FP32/20/Euler/cfg1; same skip-smooth postprocess, no algorithm edits.
User frames are Blender first=1. Both exported files have identical fingers/betas.
"""
from pathlib import Path
import sys,os,time,json,copy
import numpy as np,torch
from scipy.spatial.transform import Rotation as R,Slerp
P=Path(__file__).resolve().parents[1];REPO=P/'app/flowhmr';sys.path.insert(0,str(REPO));os.chdir(REPO)
os.environ['POSTPROCESS_SKIP_SMOOTH']='1'
from flowhmr.pipeline.v2m_inference import V2MInferenceEngine,build_feature_dict,_rotate_smplh_params_y180
from flowhmr.utils.runtime_vggt_camera import _vggt_opencv_to_yup_rt
import flowhmr.pipeline.pipeline_v2m as pm
from flowhmr.core.math.geometry import rot6d_to_rotation_matrix,rotation_matrix_to_angle_axis
from flowhmr.core.export.smplh2fbx import SMPLH2FBX
D=P/'outputs/violent_overlap_review';D.mkdir(exist_ok=True);C=P/'inputs/violent_original_cache'
# 0-based inclusive indices. Public frame numbers = index+1.
A0,A1=699,1194;B0,B1=1105,1599;S,E=799,1499;OV0,OV1=B0,A1
T_ORIG=3789
with np.load(r'F:\flowhmr_official_out\video_HaMeRhands_leftfixed.npz',allow_pickle=False) as z:ref={k:z[k].copy() for k in z.files}
with np.load(r'F:\flowhmr_official_out\video_OFFICIAL_nopost.npz',allow_pickle=False) as z:raw_ref={k:z[k].copy() for k in z.files}
tok=torch.load(C/'video_cut_1080p30_sam3d_feat.pt',map_location='cpu',weights_only=True)
noise=torch.load(C/'global_noise_seed0.pt',map_location='cpu',weights_only=True)
z=np.load(C/'video_cut_1080p30_vggt_camera.npz');cam=_vggt_opencv_to_yup_rt(torch.from_numpy(z['RT_vggt'].astype(np.float32)))
assert len(tok)==len(cam)==T_ORIG and noise.shape==(1,T_ORIG,491)
assert np.array_equal(z['frame_range'],[0,3788])
engine=V2MInferenceEngine.build_lite(str(REPO/'checkpoints/flowhmr_latest/flowhmr_latest.ckpt'),str(C/'original_checkpoint_config.yml'),device='cuda',postprocess=True)
report={'reference':r'F:\flowhmr_official_out\video_HaMeRhands_leftfixed.npz','input_video':r'F:\flowhmr_upload\video_cut_1080p30.mp4','user_frame_convention':'Blender first=1','chunk_frames_1based':{'A':[A0+1,A1+1],'B':[B0+1,B1+1]},'review_frames_1based':[S+1,E+1],'blend_frames_1based':[OV0+1,OV1+1],'review_local_blend_frames_1based':[OV0-S+1,OV1-S+1],'settings':{'FP32':True,'steps':20,'cfg':1,'seed':0,'skip_regular_3frame_smooth':True,'original_feature_camera_noise_cache':True,'original_global_noise_generated_on_cloud_cpu':True,'independent_chunk_postprocess_same_as_reference':True,'hands_identical_to_user_reference':True,'export_betas_shared_user_reference':True,'no_full_sequence_generation_rerun':True},'chunks':{}}
def st(v):return {'mean':float(np.mean(v)),'p99':float(np.percentile(v,99)),'max':float(np.max(v))}
def geodesic(a,b):return np.degrees((R.from_rotvec(a.reshape(-1,3)).inv()*R.from_rotvec(b.reshape(-1,3))).magnitude()).reshape(-1,52)
def serialize(out):
    p=rotation_matrix_to_angle_axis(rot6d_to_rotation_matrix(out['rot6d'][0])).detach().cpu().numpy().copy()
    t=out['trans'][0].detach().cpu().numpy().copy();_rotate_smplh_params_y180(p,t)
    return {'poses':p,'trans':t,'betas':out['shapes'][0,:1].detach().cpu().numpy().copy()}
results={};original=pm.randn_tensor
for tag,start,end in [('A',A0,A1),('B',B0,B1)]:
    n=end-start+1;calls=[]
    def fixed(shape,generator=None,device=None,dtype=None,layout=None):
        assert tuple(shape)==(1,n,491);calls.append(1)
        return noise[:,start:end+1].clone().to(device=device or 'cpu',dtype=dtype or torch.float32)
    pm.randn_tensor=fixed;t0=time.perf_counter();torch.cuda.reset_peak_memory_stats()
    try:
        feat=build_feature_dict(tok[start:end+1],cam[start:end+1],engine.device)
        with torch.no_grad():out=engine.pipeline.generate(feature=feat,seeds=[0],length=n,cfg_scale=1.)
        assert calls==[1]
        raw=serialize(out);np.savez(D/f'chunk_{tag}_raw.npz',**raw)
        out=engine.postprocess.run(out);post=serialize(out);np.savez(D/f'chunk_{tag}_post.npz',**post)
        results[tag]=post
        report['chunks'][tag]={'frames':n,'seconds':time.perf_counter()-t0,'peak_alloc_GiB':torch.cuda.max_memory_allocated()/2**30,'raw_local_body_difference_vs_old_whole_raw_deg':st(geodesic(raw_ref['poses'][start:end+1],raw['poses'])[:,1:22])}
        print('CHUNK_DONE',tag,report['chunks'][tag],flush=True)
        del out,feat;torch.cuda.empty_cache()
    finally:pm.randn_tensor=original

A,B=results['A'],results['B'];nOV=OV1-OV0+1;assert nOV==90
ia=OV0-A0
# Estimate ONE rigid world alignment per chunk; never per-frame warp to control.
def alignment(pa,ta,pb,tb):
    rel=R.from_rotvec(pa[:,0])*R.from_rotvec(pb[:,0]).inv()
    yaw=float(np.median(rel.as_euler('yxz')[:,0]));r=R.from_euler('y',yaw)
    off=ta.mean(0)-r.apply(tb).mean(0);return r,off,yaw
rB,offB,yB=alignment(A['poses'][ia:ia+nOV],A['trans'][ia:ia+nOV],B['poses'][:nOV],B['trans'][:nOV])
bp=B['poses'].astype(np.float64).copy();bp[:,0]=(rB*R.from_rotvec(bp[:,0])).as_rotvec();bt=rB.apply(B['trans'])+offB
w=np.linspace(0,1,nOV);mid=A['poses'][ia:ia+nOV].astype(np.float64).copy()
for i in range(nOV):
    for j in range(52):mid[i,j]=Slerp([0,1],R.from_rotvec(np.stack([mid[i,j],bp[i,j]])))([w[i]]).as_rotvec()[0]
p=np.concatenate([A['poses'][:ia],mid,bp[nOV:]]);t=np.concatenate([A['trans'][:ia],(1-w[:,None])*A['trans'][ia:ia+nOV]+w[:,None]*bt[:nOV],bt[nOV:]])
assert len(p)==B1-A0+1
# Align the assembled motion ONCE using a prefix BEFORE the blend.
fit0,fit1=S,999
sl=slice(fit0-A0,fit1-A0+1);r0,off0,y0=alignment(ref['poses'][fit0:fit1+1],ref['trans'][fit0:fit1+1],p[sl],t[sl])
p[:,0]=(r0*R.from_rotvec(p[:,0])).as_rotvec();t=r0.apply(t)+off0
p=p[S-A0:E-A0+1];t=t[S-A0:E-A0+1]
# Ignore hand motion completely by copying reference fingers verbatim into BOTH.
p[:,22:]=ref['poses'][S:E+1,22:]
reference={'poses':ref['poses'][S:E+1].copy(),'trans':ref['trans'][S:E+1].copy(),'betas':ref['betas'].copy()}
experiment={'poses':p.astype(np.float32),'trans':t.astype(np.float32),'betas':ref['betas'].copy()}
assert np.array_equal(experiment['poses'][:,22:],reference['poses'][:,22:])
exports=[]
for name,dat in [('V00_original_whole_crop_0800_1500',reference),('V01_controlled_overlap_crossfade_0800_1500',experiment)]:
    dat.update(Rh=dat['poses'][:,0].copy(),mocap_framerate=30,num_frames=len(dat['poses']),gender='neutral')
    npz=D/f'{name}.npz';fbx=D/f'{name}.fbx';np.savez(npz,**dat)
    conv=SMPLH2FBX(str(REPO/'assets/body_models/smplh/neutral/model.npz'),obj_template_path=None);assert conv.convert_npz_to_fbx(str(npz),str(fbx));conv.mgr.Destroy()
    exports.append({'name':name,'npz':str(npz),'fbx':str(fbx),'frames':len(dat['poses']),'bytes':fbx.stat().st_size});print('EXPORT',name,flush=True)
err=geodesic(reference['poses'],experiment['poses']);trans_err=np.linalg.norm(reference['trans']-experiment['trans'],axis=1)*100
report['metrics']={'body_local_ex_root_deg':st(err[:,1:22]),'root_orientation_deg':st(err[:,0]),'root_trajectory_cm_after_single_prefix_alignment':st(trans_err),'violent_0928_1370_body_local_deg':st(err[927-S:1370-S,1:22]),'overlap_body_local_deg':st(err[OV0-S:OV1-S+1,1:22])}
report['alignment']={'B_to_A_yaw_deg':float(np.degrees(yB)),'B_to_A_translation_cm':(offB*100).tolist(),'assembled_to_reference_yaw_deg':float(np.degrees(y0)),'assembled_to_reference_translation_cm':(off0*100).tolist(),'alignment_reference_frames_1based':[fit0+1,fit1+1],'one_global_rigid_transform_only':True}
report['exports']=exports;report['notes']=['Local vs original cloud environment can also contribute numerical differences.','Postprocessing runs independently per chunk, as it did for the whole reference; no rolling/ground algorithms changed.','Reference crop is copied, not regenerated. Fingers and export betas identical so compare body/root only.']
(D/'experiment.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8');print('DONE',json.dumps(report),flush=True)
