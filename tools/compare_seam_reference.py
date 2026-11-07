from pathlib import Path
import json,numpy as np
from scipy.spatial.transform import Rotation as R
D=Path(__file__).resolve().parents[1]/'outputs/manual_review';m=json.loads((D/'review_manifest.json').read_text());full=np.load(D/'C00_whole.npz');rows=[]
def step(z,i):
    p=z['poses'];t=z['trans'];d=np.degrees((R.from_rotvec(p[i-1]).inv()*R.from_rotvec(p[i])).magnitude())
    return {'root_cm':float(np.linalg.norm(t[i]-t[i-1])*100),'body_max_deg':float(d[:22].max())}
for name in ['C01_legacy_no_overlap','C02_legacy_overlap_hardcut','C03_overlap_slerp_crossfade','C04_no_overlap_root_aligned']:
    z=np.load(D/f'{name}.npz')
    for q in m['metrics'][name]['seams']:
        i=q['frame0'];a=step(z,i);b=step(full,i);rows.append({'name':name,'frame0':i,'case':a,'same_frame_whole':b,'root_magnitude_ratio':a['root_cm']/b['root_cm'],'body_max_magnitude_ratio':a['body_max_deg']/b['body_max_deg']})
(D/'same_frame_seam_reference.json').write_text(json.dumps(rows,indent=2),encoding='utf-8');print(json.dumps(rows,indent=2))
