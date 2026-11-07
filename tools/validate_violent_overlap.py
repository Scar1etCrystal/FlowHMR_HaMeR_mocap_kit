from pathlib import Path
import json,sys,numpy as np,fbx
from scipy.spatial.transform import Rotation as R
P=Path(__file__).resolve().parents[1];D=P/'outputs/violent_overlap_review';sys.path.insert(0,str(P/'app/flowhmr'))
from flowhmr.core.export.fbx_utils import SMPLH_JOINT2NUM
rep=json.loads((D/'experiment.json').read_text());rows=[]
for it in rep['exports']:
    z=np.load(it['npz'],allow_pickle=False);T=len(z['poses']);m=fbx.FbxManager.Create();m.SetIOSettings(fbx.FbxIOSettings.Create(m,fbx.IOSROOT));imp=fbx.FbxImporter.Create(m,'');assert imp.Initialize(it['fbx'],-1,m.GetIOSettings());s=fbx.FbxScene.Create(m,'check');assert imp.Import(s);imp.Destroy();nodes={};meshes=[]
    def walk(n):
        a=n.GetNodeAttribute()
        if a and a.GetAttributeType()==fbx.FbxNodeAttribute.EType.eSkeleton:nodes[n.GetName()]=n
        if a and a.GetAttributeType()==fbx.FbxNodeAttribute.EType.eMesh:meshes.append(n)
        for i in range(n.GetChildCount()):walk(n.GetChild(i))
    walk(s.GetRootNode());assert len(nodes)==52 and len(meshes)==1
    stack=s.GetSrcObject(fbx.FbxCriteria.ObjectType(fbx.FbxAnimStack.ClassId),0);layer=stack.GetSrcObject(fbx.FbxCriteria.ObjectType(fbx.FbxAnimLayer.ClassId),0);angles=np.empty((T,52,3))
    for name,j in SMPLH_JOINT2NUM.items():
        for k,axis in enumerate('XYZ'):
            c=nodes[name].LclRotation.GetCurve(layer,axis);assert c.KeyGetCount()==T;assert abs(c.KeyGetTime(T-1).GetSecondDouble()-(T-1)/30)<1e-7
            angles[:,j,k]=[c.KeyGetValue(i) for i in range(T)]
    e=np.degrees((R.from_euler('xyz',angles.reshape(-1,3),degrees=True).inv()*R.from_rotvec(z['poses'].reshape(-1,3))).magnitude());assert e.max()<.005
    r=np.stack([[nodes['Pelvis'].LclTranslation.GetCurve(layer,a).KeyGetValue(i) for i in range(T)] for a in 'XYZ'],axis=-1);te=np.max(np.abs((r-r[0])-(z['trans']-z['trans'][0])*100));assert te<.01
    rows.append({'name':it['name'],'frames':T,'rotation_key_max_error_deg':float(e.max()),'root_key_max_error_cm':float(te),'skeletons':52,'meshes':1});m.Destroy()
a=np.load(rep['exports'][0]['npz']);b=np.load(rep['exports'][1]['npz']);assert np.array_equal(a['poses'][:,22:],b['poses'][:,22:]);assert np.array_equal(a['betas'],b['betas'])
original=np.load(rep['reference']);assert np.array_equal(a['poses'],original['poses'][799:1500]);assert np.array_equal(a['trans'],original['trans'][799:1500])
rep['fbx_validation']=rows;rep['reference_crop_exact']=True;rep['hands_betas_same']=True
(D/'experiment.json').write_text(json.dumps(rep,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(rows,indent=2))
text='''# 剧烈运动区：只做受控重叠渐变

先打开 V00_original_whole_crop_0800_1500.fbx（原版），再打开 V01_controlled_overlap_crossfade_0800_1500.fbx（实验）。

两份都是701帧、30fps，首帧映射到原FBX的第800帧；最后一帧映射第1500帧。若Blender导入首帧为1：

- 你指定的剧烈运动928–1370：本文件第129–571帧。
- 90帧重叠渐变1106–1195：本文件第307–396帧。
- 渐变中部约原第1150/1151帧：本文件第351/352帧。
- **没有单一硬切接缝**：检查整段307–396帧，以及进入/退出渐变的306→307、396→397。

源视频：F:\\flowhmr_upload\\video_cut_1080p30.mp4。
原版：用户指定video_HaMeRhands_leftfixed.npz直接裁剪，身体不重跑。
块A原700–1195、块B1106–1600，沿用原全片SAM特征、相机、seed0全局噪声切片；FP32、20步、CFG1。
各块同原去滤波后处理，然后重叠根yaw/XYZ一次对齐，再90帧旋转SLERP/平移渐变。整段仅用渐变之前800–1000做一次刚体对齐，未逐帧追原轨迹。
两份使用同一原版体型、同一原版30个指关节，忽略手部；根和身体0..21供比较。
本机与旧云端环境差异、各块后处理上下文也会进入最终差异，本实验不声称纯注意力窗口因果隔离。
不改地面算法，不恢复此前已搁置的打滚排查。

'''
met=rep['metrics'];text+=f"简单数值：身体局部关节1..21平均差{met['body_local_ex_root_deg']['mean']:.2f}°，最大{met['body_local_ex_root_deg']['max']:.2f}°；剧烈区平均差{met['violent_0928_1370_body_local_deg']['mean']:.2f}°。根轨迹平均差{met['root_trajectory_cm_after_single_prefix_alignment']['mean']:.2f}cm。以人工观感为主。\n\nFBX已重新导入核对52骨骼、1网格、701帧以及每帧旋转/根轨迹与NPZ一致；未做Blender视觉判断。\n"
(D/'README_检查位置.md').write_text(text,encoding='utf-8')
