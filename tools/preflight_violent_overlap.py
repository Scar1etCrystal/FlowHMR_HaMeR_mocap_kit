from pathlib import Path
import json,numpy as np,cv2,fbx,hashlib,sys
P=Path(__file__).resolve().parents[1];D=P/'outputs/violent_overlap_review';D.mkdir(parents=True,exist_ok=True)
video=Path(r'F:\flowhmr_upload\video_cut_1080p30.mp4');ref=Path(r'F:\flowhmr_official_out\video_HaMeRhands_leftfixed.npz');fb=ref.with_suffix('.fbx')
with np.load(ref,allow_pickle=False) as z:meta={k:{'shape':list(z[k].shape),'dtype':str(z[k].dtype)} for k in z.files};poses=z['poses'].copy();trans=z['trans'].copy()
with np.load(r'F:\flowhmr_official_out\video_OFFICIAL_post_nosmooth.npz',allow_pickle=False) as z:
    n=min(len(poses),len(z['poses']))
    same={'body_0_21_bit_equal_in_original_range':bool(np.array_equal(poses[:n,:22],z['poses'][:n,:22])),'trans_bit_equal_in_original_range':bool(np.array_equal(trans[:n],z['trans'][:n])),'original_body_frames':len(z['poses']),'reference_frames':len(poses),'reference_pads_last_body_frame':bool(np.array_equal(poses[n:,:22],np.repeat(z['poses'][-1:,:22],len(poses)-n,axis=0)))}
cap=cv2.VideoCapture(str(video));vm={'frames':int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),'fps':cap.get(cv2.CAP_PROP_FPS),'width':int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),'height':int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))};cap.release()
m=fbx.FbxManager.Create();m.SetIOSettings(fbx.FbxIOSettings.Create(m,fbx.IOSROOT));imp=fbx.FbxImporter.Create(m,'');assert imp.Initialize(str(fb),-1,m.GetIOSettings());scene=fbx.FbxScene.Create(m,'ref');assert imp.Import(scene);imp.Destroy()
stack=scene.GetSrcObject(fbx.FbxCriteria.ObjectType(fbx.FbxAnimStack.ClassId),0);layer=stack.GetSrcObject(fbx.FbxCriteria.ObjectType(fbx.FbxAnimLayer.ClassId),0)
def find(n):
    if n.GetName()=='Pelvis':return n
    for i in range(n.GetChildCount()):
        r=find(n.GetChild(i))
        if r:return r
root=find(scene.GetRootNode());c=root.LclRotation.GetCurve(layer,'X');fm={'keys':c.KeyGetCount(),'first_key_seconds':c.KeyGetTime(0).GetSecondDouble(),'last_key_seconds':c.KeyGetTime(c.KeyGetCount()-1).GetSecondDouble()};m.Destroy()
r={'source_video':str(video),'reference_npz':str(ref),'reference_fbx':str(fb),'npz':meta,'video':vm,'fbx_time':fm,'matches_official_post_nosmooth_body':same,'frame_convention':'User frame928..1370 interpreted as Blender first1 -> data index927..1369; official range0..3788 in preserved cloud log','local_original_long_feature_cache_found':False,'local_original_long_camera_cache_found':False}
(D/'preflight.json').write_text(json.dumps(r,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(r,ensure_ascii=False,indent=2))
