"""Diagnostic: preserve N60 hand-pointing direction, transfer only palm roll.
Targets are from cached HaMeR in the confirmed fixed camera. No reference fitting.
"""
from pathlib import Path
import json,pickle,sys
import numpy as np
from scipy.spatial.transform import Rotation as R
P=Path(__file__).resolve().parents[1];sys.path.insert(0,str(P/'tools'))
import trial_hamer_wrist_merge as t
root=P/'outputs/hamer_wrist_trial/fixed_camera_control'
base=t.load(root/'N60_HaMeR_N2/final.npz');ref=t.load(root/'N1_HaMeR_N1/final.npz')
model=np.load(t.MODEL);parents=model['kintree_table'][0].astype(int)[:52];parents[0]=-1
rest=model['J_regressor']@model['v_template']
_,bw,_=t.kinematics(base['poses'],rest,parents);_,rw,_=t.kinematics(ref['poses'],rest,parents)
with (root/'N60_HaMeR_N2/hands/working_30fps_hamer_data.pkl').open('rb') as f:h=pickle.load(f)
physical=h['global_orient'].astype(float).copy();physical[:,0]=t.M@physical[:,0]@t.M
report=json.loads((root/'wrist_merge/wrist_trial_results.json').read_text(encoding='utf-8'))
B=np.array(report['rest_basis_from_neutral_models']);WV=np.array(report['WV_from_yup'])
with np.load(root/'N60_HaMeR_N2/camera.npz') as z:Rcw0=z['RT_vggt'][0,:3,:3]@t.S
tr=t.E@WV@Rcw0.T;target=tr@physical@B
valid=h['global_orient_valid'].astype(bool);rows=[]
for s,(w,index,middle,pinky) in enumerate([(20,22,25,28),(21,37,40,43)]):
    forward=rest[middle]-rest[w];forward/=np.linalg.norm(forward)
    lateral=rest[index]-rest[pinky]
    normal=np.cross(forward,lateral);normal/=np.linalg.norm(normal)
    axis=bw[:,w]@forward;n0=bw[:,w]@normal;n1=target[:,s]@normal
    n1-=np.einsum('ti,ti->t',n1,axis)[:,None]*axis
    length=np.linalg.norm(n1,axis=-1);v=valid[:,s]&(length>.1)
    n1/=np.maximum(length[:,None],1e-8)
    signed=np.arctan2(np.einsum('ti,ti->t',axis,np.cross(n0,n1)),np.einsum('ti,ti->t',n0,n1))
    corrected=R.from_rotvec(axis*signed[:,None]).as_matrix()@bw[:,w]
    rows.append({'side':s,'before':t.stats(t.angles(bw[v,w],rw[v,w])),
                 'palm_roll_only':t.stats(t.angles(corrected[v],rw[v,w])),
                 'valid_frames':int(v.sum())})
(root/'palm_roll_diagnostic.json').write_text(json.dumps(rows,indent=2),encoding='utf-8');print(json.dumps(rows,indent=2))
