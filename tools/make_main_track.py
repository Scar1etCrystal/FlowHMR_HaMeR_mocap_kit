"""Convert FlowHMR bbox cache to the hand runner's frame-aligned track."""
from pathlib import Path
import argparse,numpy as np
ap=argparse.ArgumentParser(description=__doc__)
ap.add_argument('--bbox',required=True,help='FlowHMR *_bbox.npz')
ap.add_argument('--output',required=True,help='Output main_track.npz')
a=ap.parse_args()
with np.load(a.bbox,allow_pickle=False) as z:
    boxes=z['bbox'].astype(np.float32)
    # The FlowHMR cache has no observed-vs-interpolated detection provenance.
    valid=np.isfinite(boxes).all(axis=1)&(boxes[:,2]>boxes[:,0])&(boxes[:,3]>boxes[:,1])
    d={'bbox_xyxy':boxes,'detected':valid,'detected_semantics':np.array('valid bbox; original detector provenance unavailable')}
    if 'start_end' in z:d['start_end']=z['start_end'].copy()
p=Path(a.output);p.parent.mkdir(parents=True,exist_ok=True);np.savez(p,**d)
print('TRACK',boxes.shape,str(p),'detected indicates valid cache box, not original YOLO observation')
