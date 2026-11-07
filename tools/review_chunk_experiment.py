"""Audit stored chunk experiment and export an explicitly labelled review set.
CPU only. Does not rerun models or alter production sources/original results.
"""
from pathlib import Path
import os,sys,json,shutil,pickle
import numpy as np
from scipy.spatial.transform import Rotation as R,Slerp
import cv2,torch
P=Path(__file__).resolve().parents[1];REPO=P/'app/flowhmr';sys.path.insert(0,str(REPO));os.chdir(REPO)
from flowhmr.core.export.smplh2fbx import SMPLH2FBX
from flowhmr.pipeline.utils import randn_tensor
O=REPO/'output';D=P/'outputs/manual_review';D.mkdir(parents=True,exist_ok=True)
SPECS={'truncA':(0,150),'truncB':(150,150),'overA':(0,195),'overB':(105,195)}
def load(tag):
    with np.load(O/f'chunk_{tag}/flowhmr_latest.npz',allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def rotations(p):return R.from_rotvec(np.asarray(p).reshape(-1,3))
def geo(a,b):return np.degrees((rotations(a).inv()*rotations(b)).magnitude()).reshape(-1,52)
def stats(a):return {'mean':float(np.mean(a)),'p99':float(np.percentile(a,99)),'max':float(np.max(a))}
def yaw(a,b):return float(np.median((rotations(a)*rotations(b).inv()).as_euler('yxz')[:,0]))
def data(poses,trans,betas):
    return {'poses':np.asarray(poses,np.float32),'trans':np.asarray(trans,np.float32),'betas':np.asarray(betas,np.float32),'Rh':np.asarray(poses[:,0],np.float32),'mocap_framerate':np.array(30),'num_frames':np.array(len(poses)),'gender':np.array('neutral')}
manifest=[]
def export(name,z,description):
    npz=D/f'{name}.npz';fb=D/f'{name}.fbx';np.savez(npz,**z)
    c=SMPLH2FBX(str(REPO/'assets/body_models/smplh/neutral/model.npz'),obj_template_path=None)
    assert c.convert_npz_to_fbx(str(npz),str(fb),export_mesh=True)
    manifest.append({'name':name,'description':description,'frames':len(z['poses']),'fbx_bytes':fb.stat().st_size,'npz':str(npz),'fbx':str(fb)})
    c.mgr.Destroy();print('EXPORTED',name,len(z['poses']),fb.stat().st_size,flush=True)

# H group: EXACT same stored body for N1/N2/N3. Fingers only, same mirror and exporter.
with np.load(O/'example/flowhmr_latest.npz',allow_pickle=False) as z:body={k:z[k].copy() for k in z.files}
export('H00_flowhmr_body_only',data(body['poses'],body['trans'],body['betas']),'151 frames; no HaMeR replacement')
M=np.diag([-1.,1.,1.])
for N in (1,2,3):
    with (P/f'outputs/hamer_pose_n{N}/example_30fps_hamer_data.pkl').open('rb') as f:h=pickle.load(f)
    poses=body['poses'].copy();hp=np.asarray(h['hand_pose']);valid=np.asarray(h['global_orient_valid'],bool)
    assert len(poses)==len(hp)
    for side,start in ((0,22),(1,37)):
        mats=hp[valid[:,side],side]
        if side==0:mats=M@mats@M
        poses[valid[:,side],start:start+15]=R.from_matrix(mats.reshape(-1,3,3)).as_rotvec().reshape(-1,15,3)
    assert np.array_equal(poses[:,:22],body['poses'][:,:22])
    export(f'H0{N}_hamer_N{N}',data(poses,body['trans'],body['betas']),f'151 frames; identical body/root/wrists, HaMeR N={N} fingers only')

# C group: stored original experiment, not a generation rerun.
z={tag:load(tag) for tag in ['full',*SPECS]};F=z['full'];A=z['truncA'];B=z['truncB'];OA=z['overA'];OB=z['overB']
assert [len(z[t]['poses']) for t in z]==[300,150,150,195,195]
rawP=np.concatenate([A['poses'],B['poses']]);rawT=np.concatenate([A['trans'],B['trans']])
y=yaw(OA['poses'][105:195,0],OB['poses'][:90,0]);ry=R.from_euler('y',y)
bt=ry.apply(OB['trans']-OB['trans'][:90].mean(0))+OA['trans'][105:195].mean(0)
bp=OB['poses'].astype(np.float64).copy();bp[:,0]=(ry*rotations(bp[:,0])).as_rotvec()
cut=149
hardP=np.concatenate([OA['poses'][:cut],bp[cut-105:]]);hardT=np.concatenate([OA['trans'][:cut],bt[cut-105:]])
# Additional demonstration: ONLY replace the legacy hard cut with a 90-frame crossfade.
a=np.linspace(0,1,90);blendP=OA['poses'][105:195].astype(np.float64).copy()
for t in range(90):
    for j in range(52):blendP[t,j]=Slerp([0,1],R.from_rotvec(np.stack([OA['poses'][105+t,j],bp[t,j]])))([a[t]]).as_rotvec()[0]
blendT=(1-a[:,None])*OA['trans'][105:195]+a[:,None]*bt[:90]
blendP=np.concatenate([OA['poses'][:105],blendP,bp[90:]])
blendT=np.concatenate([OA['trans'][:105],blendT,bt[90:]])
# No-overlap can be aligned too. It is an approximation, not a new control experiment.
ay=yaw(A['poses'][-1:,0],B['poses'][:1,0]);ar=R.from_euler('y',ay)
ap=B['poses'].astype(np.float64).copy();ap[:,0]=(ar*rotations(ap[:,0])).as_rotvec()
pred=A['trans'][-1]+(A['trans'][-1]-A['trans'][-2]);at=ar.apply(B['trans'])+pred-ar.apply(B['trans'][0])
alignedP=np.concatenate([A['poses'],ap]);alignedT=np.concatenate([A['trans'],at])
cases={'C00_whole':(F['poses'],F['trans'],'original local whole-video reference'),
       'C01_legacy_no_overlap':(rawP,rawT,'exact legacy naive concatenation; seam global frame150'),
       'C02_legacy_overlap_hardcut':(hardP,hardT,'exact legacy yaw/XYZ align and hard switch at global frame149; NOT blending'),
       'C03_overlap_slerp_crossfade':(blendP,blendT,'NEW demonstration only: same legacy inputs/alignment, SLERP/translation crossfade global105..194'),
       'C04_no_overlap_root_aligned':(alignedP,alignedT,'NEW demonstration only: seam root yaw and extrapolated position aligned; not proof of accuracy')}
metrics={}
def seam(p,t,index):return {'frame0':index,'root_step_cm':float(np.linalg.norm(t[index]-t[index-1])*100),'body_joint_step_max_deg':float(geo(p[index-1:index],p[index:index+1])[:,:22].max())}
for name,(p,t,desc) in cases.items():
    export(name,data(p,t,F['betas']),desc+'; shared whole-reference betas for all assembled C exports')
    metrics[name]={'body_local_ex_root_deg':stats(geo(F['poses'],p)[:,1:22]),'seams':[seam(p,t,k) for k in ([105,195] if 'crossfade' in name else [149] if 'hardcut' in name else [150])],'max_body_step_any_frame_deg':float(geo(p[:-1],p[1:])[:,:22].max())}
for i,tag in enumerate(SPECS,10):export(f'C{i}_{tag}_standalone',data(z[tag]['poses'],z[tag]['trans'],z[tag]['betas']),f'original independent chunk {tag}; original own betas/world frame; do not overlay without alignment')

# Preprocessing audit: frame alignment and pixel perturbation after re-encoding.
def smallframes(path):
    c=cv2.VideoCapture(str(path));fps=c.get(cv2.CAP_PROP_FPS);f=[]
    while True:
        ok,img=c.read()
        if not ok:break
        f.append(cv2.resize(img,(96,96),interpolation=cv2.INTER_AREA))
    c.release();return np.stack(f),fps
fv,ffps=smallframes(O/'chunk_full/chunk_full_30fps.mp4')
ft=torch.load(O/'chunk_full/chunk_full_sam3d_feat.pt',map_location='cpu',weights_only=True)
fullbox=np.load(O/'chunk_full/chunk_full_bbox.npz')['bbox'];fullcam=dict(np.load(O/'chunk_full/chunk_full_vggt_camera.npz'))
noise=lambda T:randn_tensor((1,T,491),generator=torch.Generator().manual_seed(0),dtype=torch.float32)[0]
fullnoise=noise(300);audit={}
shape=np.load(REPO/'assets/body_models/smplh/neutral/model.npz',allow_pickle=True)['shapedirs']
for tag,(start,n) in SPECS.items():
    base=O/f'chunk_{tag}';v,fps=smallframes(base/f'chunk_{tag}_30fps.mp4');assert len(v)==n
    ref=fv[start:start+n]
    shifts={}
    for shift in range(-2,3):
        lo=max(0,-start-shift);hi=min(n,len(fv)-start-shift)
        shifts[shift]=float(np.abs(v[lo:hi].astype(float)-fv[start+shift+lo:start+shift+hi].astype(float)).mean())
    b=np.load(base/f'chunk_{tag}_bbox.npz')['bbox'];fb=fullbox[start:start+n]
    mn=np.maximum(b[:,:2],fb[:,:2]);mx=np.minimum(b[:,2:],fb[:,2:]);inter=np.prod(np.maximum(0,mx-mn),axis=1)
    area=lambda q:np.prod(np.maximum(0,q[:,2:]-q[:,:2]),axis=1)
    iou=inter/np.maximum(area(b)+area(fb)-inter,1e-6)
    f=torch.load(base/f'chunk_{tag}_sam3d_feat.pt',map_location='cpu',weights_only=True)
    cs={name:stats(torch.nn.functional.cosine_similarity(ft[start:start+n,sl],f[:,sl],dim=1).numpy()) for name,sl in [('all',slice(None)),('body',slice(0,1024)),('left',slice(1024,2048)),('right',slice(2048,3072))]}
    cam=dict(np.load(base/f'chunk_{tag}_vggt_camera.npz'));K=np.asarray(cam['K']);FK=np.asarray(fullcam['K'])[start:start+n]
    rc=R.from_matrix(cam['RT_vggt'][:,:3,:3]);rf=R.from_matrix(fullcam['RT_vggt'][start:start+n,:3,:3])
    cd=np.degrees(((rf*rf[0].inv()).inv()*(rc*rc[0].inv())).magnitude())
    db=(z[tag]['betas']-F['betas']).reshape(-1);meshdelta=np.einsum('vcn,n->vc',shape[:,:,:len(db)],db)
    nn=noise(n);nf=fullnoise[start:start+n]
    audit[tag]={'frames':len(v),'fps':fps,'global_range_inclusive':[start,start+n-1],'small_rgb_pixel_MAE_by_shift':shifts,'best_shift':min(shifts,key=shifts.get),'bbox_iou_mean':float(iou.mean()),'bbox_iou_min':float(iou.min()),'bbox_exact_equal':bool(np.array_equal(b,fb)),'feature_cos':cs,'K_focal_relative_difference_mean':float(np.mean(np.abs(K[:,[0,1],[0,1]]-FK[:,[0,1],[0,1]])/np.maximum(np.abs(FK[:,[0,1],[0,1]]),1e-6))),'camera_relative_rotation_deg':stats(cd),'beta_diff_L2':float(np.linalg.norm(db)),'neutral_shape_vertex_difference_mm':stats(np.linalg.norm(meshdelta,axis=-1)*1000),'noise_same_scalar_fraction':float((nn==nf).float().mean()),'noise_RMS_difference':float(torch.sqrt(((nn-nf)**2).mean())),'camera_sample_global_frames':list(range(start,start+n,10))+([] if (n-1)%10==0 else [start+n-1])}

report={'export_manifest':manifest,'audit':audit,'metrics':metrics,'legacy_overlap_yaw_deg':float(np.degrees(y)),'legacy_cut_frame0':149,'shared_betas_for_assembled_exports':True,'no_model_rerun':True,'warning':'Original independent pipeline comparison, NOT a controlled generation-only experiment and NOT cloud-controlled. C group does not contain HaMeR.'}
(D/'review_manifest.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
shutil.copy2(O/'example/example_30fps.mp4',D/'H_source_30fps.mp4');shutil.copy2(O/'chunk_full/chunk_full_30fps.mp4',D/'C_source_30fps.mp4')
print('AUDIT',json.dumps(audit),flush=True);print('METRICS',json.dumps(metrics),flush=True)
