"""Diagnose constant frame discrepancies; do not select a fitted correction for export."""
from pathlib import Path
import sys,pickle,json
import numpy as np
from scipy.spatial.transform import Rotation as R
P=Path(__file__).resolve().parents[1];sys.path.insert(0,str(P/'tools'))
# Import audit helpers (also refreshes its pure-CPU report).
import audit_hamer_wrist_coordinates as a
H=a.phys;B=a.basis;valid=a.valid;ref=a.cw[:,[20,21]]
cv=a.camera['RT_vggt'][:,:3,:3]
I=np.eye(3);axes={'I':I,'Rx180':a.S,'Ry180':a.E,'Rz180':a.E@a.S}
rows=[]
for leftname,left in axes.items():
    for rightname,right in axes.items():
        predicted=left@np.swapaxes(cv,1,2)[:,None]@H@right@B
        rows.append({'left':leftname,'right':rightname,'mean_by_side':[float(a.angles(predicted[:,s][valid[:,s]],ref[:,s][valid[:,s]]).mean()) for s in [0,1]]})
rows.sort(key=lambda r:sum(r['mean_by_side']))
print('RAW_FIRST_CAMERA',cv[0].tolist())
print('BEST_DISCRETE_DIAGNOSTICS_ONLY',json.dumps(rows[:8],indent=2))
# Oracle constant left rotation: ONLY to reveal whether an unexplained global
# gauge remains. This is never used in the actual wrist replacement.
res=[]
for side in [0,1]:
    v=valid[:,side];target=H[v,side]@B[side]
    delta=ref[v,side]@np.swapaxes(target,1,2)
    rot=R.from_matrix(delta).mean().as_matrix()
    error=a.angles(rot@target,ref[v,side])
    res.append({'side':side,'diagnostic_oracle_left_matrix':rot.tolist(),'residual_mean':float(error.mean()),'rotation_degrees':float(np.degrees(R.from_matrix(rot).magnitude()))})
print('DIAGNOSTIC_ORACLE_NOT_FOR_EXPORT',json.dumps(res,indent=2))
(P/'outputs/hamer_wrist_trial/frame_diagnostics.json').write_text(json.dumps({'discrete_diagnostics':rows,'oracle_diagnostic_not_used':res},indent=2),encoding='utf-8')
