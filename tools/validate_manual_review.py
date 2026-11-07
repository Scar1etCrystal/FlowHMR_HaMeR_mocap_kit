from pathlib import Path
import json,numpy as np,fbx
from scipy.spatial.transform import Rotation as R
P=Path(__file__).resolve().parents[1];D=P/'outputs/manual_review'
a=json.loads((D/'review_manifest.json').read_text());b=json.loads((D/'controlled_generation_audit.json').read_text());items=a['export_manifest']+b['exports'];result=[]
for item in items:
    z=np.load(item['npz'],allow_pickle=False);T=len(z['poses'])
    m=fbx.FbxManager.Create();m.SetIOSettings(fbx.FbxIOSettings.Create(m,fbx.IOSROOT));imp=fbx.FbxImporter.Create(m,'');assert imp.Initialize(item['fbx'],-1,m.GetIOSettings());s=fbx.FbxScene.Create(m,'review');assert imp.Import(s);imp.Destroy()
    nodes={};meshes=[]
    def walk(n):
        attr=n.GetNodeAttribute()
        if attr and attr.GetAttributeType()==fbx.FbxNodeAttribute.EType.eSkeleton:nodes[n.GetName()]=n
        if attr and attr.GetAttributeType()==fbx.FbxNodeAttribute.EType.eMesh:meshes.append(n)
        for i in range(n.GetChildCount()):walk(n.GetChild(i))
    walk(s.GetRootNode());assert len(nodes)==52 and len(meshes)==1,(item['name'],len(nodes),len(meshes))
    stack=s.GetSrcObject(fbx.FbxCriteria.ObjectType(fbx.FbxAnimStack.ClassId),0)
    layer=stack.GetSrcObject(fbx.FbxCriteria.ObjectType(fbx.FbxAnimLayer.ClassId),0)
    import sys
    sys.path.insert(0,str(P/'app/flowhmr'))
    from flowhmr.core.export.fbx_utils import SMPLH_JOINT2NUM
    angles=np.empty((T,52,3),float);counts=[]
    for name,j in SMPLH_JOINT2NUM.items():
        for axis,letter in enumerate('XYZ'):
            c=nodes[name].LclRotation.GetCurve(layer,letter);assert c.KeyGetCount()==T
            assert abs(c.KeyGetTime(T-1).GetSecondDouble()-(T-1)/30)<1e-7
            angles[:,j,axis]=[c.KeyGetValue(i) for i in range(T)]
    rr=R.from_euler('xyz',angles.reshape(-1,3),degrees=True);truth=R.from_rotvec(z['poses'].reshape(-1,3))
    err=np.degrees((rr.inv()*truth).magnitude());assert err.max()<.005,(item['name'],err.max())
    root=np.stack([[nodes['Pelvis'].LclTranslation.GetCurve(layer,letter).KeyGetValue(i) for i in range(T)] for letter in 'XYZ'],axis=-1)
    te=np.abs((root-root[0])-(z['trans']-z['trans'][0])*100);assert te.max()<.01,(item['name'],te.max())
    row={'name':item['name'],'frames':T,'skeletons':len(nodes),'meshes':len(meshes),'rotation_key_max_error_deg':float(err.max()),'root_trajectory_max_error_cm':float(te.max())};result.append(row);print('VALID',row,flush=True);m.Destroy()
for N in (1,2,3):
    h0=np.load(D/'H00_flowhmr_body_only.npz');h=np.load(D/f'H0{N}_hamer_N{N}.npz')
    assert np.array_equal(h0['poses'][:,:22],h['poses'][:,:22]);assert np.array_equal(h0['trans'],h['trans']);assert np.array_equal(h0['betas'],h['betas'])
(D/'fbx_validation.json').write_text(json.dumps({'files':result,'all_rotation_keys_match_npz':True,'H_body_root_wrist_identical':True,'note':'Checks integer-frame keys, not Blender importer behavior between keys.'},indent=2),encoding='utf-8')
