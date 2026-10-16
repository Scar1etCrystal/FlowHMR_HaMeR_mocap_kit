"""Locked HaMeR: run ViTPose and HaMeR on f % N == 0 ONLY.
Interpolate RAW rotation outputs via SLERP, then apply the existing full-rate
export/smoothing. N=2 default; N=1 is the full-rate baseline. No box interpolation.
"""
from pathlib import Path
import argparse,sys,os,time,json,pickle,gc
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from scipy.interpolate import interp1d
PACK=Path(__file__).resolve().parents[1]
ROOT=PACK/'app/hamer'
os.chdir(ROOT);sys.path.insert(0,str(ROOT))
os.environ.setdefault('HAMER_THREADS','8')
import hamer_locked as H
import cv2,torch

def interpolate_tasks(tasks,T,max_gap):
    result=[];provenance=np.zeros((T,2),np.uint8)
    for side in (0,1):
        obs=sorted((x for x in tasks if x['is_right']==side),key=lambda x:x['frame_idx'])
        if not obs:continue
        idx=np.array([x['frame_idx'] for x in obs]);by={x['frame_idx']:x for x in obs}
        for f in range(T):
            if f in by:
                result.append({'frame_idx':f,'is_right':side,'results':{k:np.asarray(v).copy() for k,v in by[f]['results'].items()}});provenance[f,side]=1;continue
            hi=int(np.searchsorted(idx,f));lo=hi-1
            if lo<0 or hi>=len(idx):
                nearest=obs[0] if lo<0 else obs[-1]
                if abs(f-nearest['frame_idx'])<=max_gap:
                    result.append({'frame_idx':f,'is_right':side,'results':{k:np.asarray(v).copy() for k,v in nearest['results'].items()}});provenance[f,side]=3
                continue
            a,b=obs[lo],obs[hi];fa,fb=a['frame_idx'],b['frame_idx']
            if fb-fa>max_gap:continue
            alpha=(f-fa)/(fb-fa);r={}
            for key in ('pose','orient'):
                A=np.asarray(a['results'][key]);B=np.asarray(b['results'][key]);shape=A.shape
                out=[]
                for ma,mb in zip(A.reshape(-1,3,3),B.reshape(-1,3,3)):
                    out.append(Slerp([fa,fb],Rotation.from_matrix(np.stack([ma,mb])))([f]).as_matrix()[0])
                r[key]=np.array(out,dtype=np.float32).reshape(shape)
            for key in ('betas','cam_t'):
                r[key]=((1-alpha)*np.asarray(a['results'][key])+alpha*np.asarray(b['results'][key])).astype(np.float32)
            result.append({'frame_idx':f,'is_right':side,'results':r});provenance[f,side]=2
    return result,provenance

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--video',required=True);ap.add_argument('--main_track',required=True);ap.add_argument('--out_dir',required=True)
    ap.add_argument('--detect_interval',type=int,default=2,help='Run BOTH models only on f %% N == 0 (default 2)')
    ap.add_argument('--batch_size',type=int,default=16);ap.add_argument('--window',type=int,default=11)
    ap.add_argument('--max_frames',type=int,default=0);ap.add_argument('--max_interp_gap',type=int,default=15,help='Do not bridge a missing-hand interval longer than this many frames')
    ap.add_argument('--no_smooth',action='store_true',help='Diagnostic: export interpolated raw poses without Savgol')
    a=ap.parse_args()
    if a.detect_interval<1:ap.error('detect_interval must be >=1')
    out=Path(a.out_dir);out.mkdir(parents=True,exist_ok=True);t0=time.perf_counter()
    torch.set_num_threads(H.THREADS);cv2.setNumThreads(H.THREADS)
    dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model,cfg=H.load_hamer(H.DEFAULT_CHECKPOINT);model=model.to(dev).eval();vp=H.ViTPoseModel(dev)
    cap=cv2.VideoCapture(a.video);T=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));T=min(T,a.max_frames) if a.max_frames else T
    W,Hh=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT));fps=cap.get(cv2.CAP_PROP_FPS)
    mt=np.load(a.main_track);boxes=mt['bbox_xyxy'];assert len(boxes)>=T,(len(boxes),T)
    focal=cfg.EXTRA.FOCAL_LENGTH/cfg.MODEL.IMAGE_SIZE*max(W,Hh)
    hb=H.HamerBatcher(model,cfg,dev,focal,a.batch_size);obs=[];sampled=[];nvit=0
    torch.cuda.reset_peak_memory_stats()
    infer_start=time.perf_counter()
    for f in range(T):
        ok,frame=cap.read()
        if not ok:T=f;break
        if f%a.detect_interval:continue
        sampled.append(f)
        bb=boxes[f]
        if not np.isfinite(bb).all() or min(bb[2]-bb[0],bb[3]-bb[1])<=4:continue
        res=vp.predict_pose(frame[:,:,::-1],[np.r_[bb,1.0][None].astype(np.float32)]);nvit+=1
        if not res:continue
        for side,sl in H.HAND_SLICES:
            box=H.hand_box(res[0]['keypoints'][sl])
            if box is None:continue
            hb.add((f,side),frame,box,side)
            obs.append({'frame_idx':f,'is_right':side,'bbox':box})
    hb.flush();cap.release();torch.cuda.synchronize();infer_s=time.perf_counter()-infer_start
    for x in obs:x['results']={k:np.asarray(hb.out[(x['frame_idx'],x['is_right'])][k]).copy() for k in ('pose','orient','betas','cam_t')}
    # Save sparse raw outputs so interpolation and smoothing can be audited separately.
    with (out/'sampled_raw.pkl').open('wb') as fp:pickle.dump({'observations':obs,'T':T,'sampled_frames':sampled},fp,protocol=4)
    tasks,prov=interpolate_tasks(obs,T,a.max_interp_gap)
    window=T+1 if a.no_smooth else a.window
    ex=H.export_like_original(tasks,T,window)
    stem=Path(a.video).stem
    with (out/f'{stem}_hamer_data.pkl').open('wb') as fp:pickle.dump(ex,fp,protocol=4)
    np.savez(out/f'{stem}_sampling.npz',source=prov,sampled_frames=sampled)
    info={'frames':T,'fps':fps,'detect_interval':a.detect_interval,'vitpose_calls':nvit,'hamer_hand_calls':len(obs),'inference_seconds':infer_s,'seconds_with_load_export':time.perf_counter()-t0,'window':0 if a.no_smooth else a.window,'tta':1,'peak_alloc_GiB':torch.cuda.max_memory_allocated()/2**30,'peak_reserved_GiB':torch.cuda.max_memory_reserved()/2**30,'provenance_legend':{'0':'invalid','1':'model','2':'interpolated','3':'endpoint held'},'valid_LR':ex['global_orient_valid'].sum(axis=0).tolist()}
    (out/'run_info.json').write_text(json.dumps(info,indent=2),encoding='utf-8');print('DONE',json.dumps(info),flush=True)
if __name__=='__main__':main()
