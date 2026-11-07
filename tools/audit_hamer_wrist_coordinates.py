"""CPU-only coordinate audit before experimenting with wrist replacement.
The N1 reference is used only for evaluation, never to fit transformations.
"""
from pathlib import Path
import json,pickle,sys
import numpy as np
from scipy.spatial.transform import Rotation as R
P=Path(__file__).resolve().parents[1];ROOT=P/'outputs/video_to_fbx'
A=ROOT/'t1499_H4_4862a2c60c75';C=ROOT/'t1499_H4_ba48bb37ff41'
OUT=P/'outputs/hamer_wrist_trial';OUT.mkdir(exist_ok=True)
M=np.diag([-1.,1.,1.]);S=np.diag([1.,-1.,-1.]);E=np.diag([-1.,1.,-1.])
model=np.load(P/'app/flowhmr/assets/body_models/smplh/neutral/model.npz',allow_pickle=True)
parents=model['kintree_table'][0].astype(np.int64)[:52];parents[0]=-1

def load_npz(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def fk(pose):
    local=R.from_rotvec(pose.reshape(-1,3)).as_matrix().reshape(-1,52,3,3)
    world=local.copy()
    for j in range(1,52):world[:,j]=world[:,parents[j]]@local[:,j]
    return local,world

def stats(x):return {'mean_deg':float(np.mean(x)),'p99_deg':float(np.percentile(x,99)),'max_deg':float(np.max(x))}
def angles(a,b):return np.degrees(R.from_matrix((np.swapaxes(a,-1,-2)@b).reshape(-1,3,3)).magnitude()).reshape(a.shape[:-2])
def fit_basis(source,target):
    # Proper rotation mapping SMPL-H rest palm vectors into MANO rest vectors.
    source=source/np.linalg.norm(source,axis=-1,keepdims=True)
    target=target/np.linalg.norm(target,axis=-1,keepdims=True)
    u,s,vt=np.linalg.svd(target.T@source);correction=np.eye(3);correction[-1,-1]=np.linalg.det(u@vt)
    return u@correction@vt

rest=model['J_regressor']@model['v_template']
with (A/'hands/working_30fps_hamer_data.pkl').open('rb') as f:hand=pickle.load(f)
H=np.asarray(hand['global_orient'],float);valid=np.asarray(hand['global_orient_valid'],bool)
phys=H.copy();phys[:,0]=M@H[:,0]@M
assert np.max(np.abs(phys-hand['global_orient_abs_lr']))<1e-6
basis=[];basis_report=[]
for side,(w,ids,name) in enumerate([(20,[22,25,28,31,34],'MANO_LEFT.pkl'),(21,[37,40,43,46,49],'MANO_RIGHT.pkl')]):
    with (P/'app/hamer/_DATA/data/mano'/name).open('rb') as f:mano=pickle.load(f,encoding='latin1')
    j=mano['J_regressor']@np.asarray(mano['v_template']);mv=j[[1,4,7,10,13]]-j[0]
    sv=rest[ids]-rest[w]
    b=fit_basis(sv,mv);basis.append(b)
    errs=angles(np.tile(np.eye(3),(1,1,1)),b[None])
    basis_report.append({'side':side,'SMPL_to_MANO_rest_rotation':b.tolist(),'rotation_magnitude_deg':float(errs[0]),'source':'neutral template wrist-to-MCP directions; no motion/reference fitting'})
basis=np.stack(basis)
camera=load_npz(A/'camera.npz');Rcw=camera['RT_vggt'][:,:3,:3]@S
WfromCam=E@np.swapaxes(Rcw,1,2)
target=WfromCam[:,None]@phys@basis[None]
a=load_npz(A/'final.npz');c=load_npz(C/'final.npz');_,aw=fk(a['poses']);_,cw=fk(c['poses'])
rows={}
for side in (0,1):
    v=valid[:,side];w=20+side
    rows[str(side)]={'original_N60_vs_N1_world':stats(angles(aw[v,w],cw[v,w])),
        'derived_HaMeR_vs_N1_world':stats(angles(target[v,side],cw[v,w])),
        'HaMeR_vs_original_N60_world':stats(angles(target[v,side],aw[v,w])),
        'valid_frames':int(v.sum())}
np.savez(OUT/'coordinate_targets.npz',target_world=target,valid=valid,basis=basis,world_from_camera=WfromCam,hamer_physical_camera=phys)
report={'left_mirror_once_verified':True,'rest_basis':basis_report,'evaluation_vs_N1_not_ground_truth':rows,
        'transform':'R_export_y180 @ inverse(R_vggt_cv @ diag(1,-1,-1)) @ R_hamer_physical @ R_rest_smpl_to_mano',
        'reference_not_used_for_fitting':True}
(OUT/'coordinate_audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report,indent=2))
