"""Replace only 30 SMPL-H finger joints with locked HaMeR MANO rotations.
No root/body/wrist changes. Left mirror applied ONCE. Frame counts must match.
"""
from pathlib import Path
import argparse,pickle,sys
import numpy as np
from scipy.spatial.transform import Rotation
P=Path(__file__).resolve().parents[1];R=P/'app/flowhmr'
def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--body',required=True,help='Postprocessed FlowHMR NPZ (same 30fps video as hands)')
    ap.add_argument('--hands',required=True,help='HaMeR hand_pose PKL')
    ap.add_argument('--output',required=True,help='Output FBX; same-stem NPZ is also saved')
    ap.add_argument('--model',default=str(R/'assets/body_models/smplh/neutral/model.npz'))
    ap.add_argument('--no_mesh',action='store_true')
    a=ap.parse_args();out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
    with np.load(a.body,allow_pickle=False) as z:d={k:z[k].copy() for k in z.files}
    with open(a.hands,'rb') as f:h=pickle.load(f)
    pose=d['poses'].reshape(-1,52,3).copy();original=pose.copy();hp=np.asarray(h['hand_pose']);valid=np.asarray(h['global_orient_valid'],bool)
    assert len(pose)==len(hp)==len(valid),f'Frame mismatch: body {len(pose)}, hands {len(hp)}; use the SAME transcoded video'
    M=np.diag([-1.,1.,1.]);counts=[]
    for side,start in [(0,22),(1,37)]:
        v=valid[:,side] & np.isfinite(hp[:,side]).all(axis=(1,2,3))
        mats=hp[v,side].copy()
        if side==0:mats=M@mats@M
        if len(mats):pose[v,start:start+15]=Rotation.from_matrix(mats.reshape(-1,3,3)).as_rotvec().reshape(-1,15,3)
        counts.append(int(v.sum()))
    assert np.array_equal(pose[:,:22],original[:,:22])
    d['poses']=pose.astype(d['poses'].dtype);npz=out.with_suffix('.npz');np.savez(npz,**d)
    with np.load(npz,allow_pickle=False) as z:
        for k in d:
            if k!='poses':assert np.array_equal(d[k],z[k]),k
    sys.path.insert(0,str(R))
    from flowhmr.core.export.smplh2fbx import SMPLH2FBX
    conv=SMPLH2FBX(smplh_model_path=a.model,obj_template_path=None)
    assert conv.convert_npz_to_fbx(str(npz),str(out),export_mesh=not a.no_mesh)
    print('MERGED',{'frames':len(pose),'valid_LR':counts,'body_wrist_root_unchanged':True,'left_mirror':'M@R@M exactly once','fbx':str(out),'bytes':out.stat().st_size})
if __name__=='__main__':main()
