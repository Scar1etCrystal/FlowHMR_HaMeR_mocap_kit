from pathlib import Path
import json,numpy as np,fbx
P=Path(__file__).resolve().parents[1];f=P/'outputs/example_merged_n2.fbx'
m=fbx.FbxManager.Create();m.SetIOSettings(fbx.FbxIOSettings.Create(m,fbx.IOSROOT));imp=fbx.FbxImporter.Create(m,'')
assert imp.Initialize(str(f),-1,m.GetIOSettings()),imp.GetStatus().GetErrorString()
s=fbx.FbxScene.Create(m,'check');assert imp.Import(s);imp.Destroy();counts={'skeleton':0,'mesh':0}
def walk(n):
    a=n.GetNodeAttribute()
    if a:
        if a.GetAttributeType()==fbx.FbxNodeAttribute.EType.eSkeleton:counts['skeleton']+=1
        elif a.GetAttributeType()==fbx.FbxNodeAttribute.EType.eMesh:counts['mesh']+=1
    for i in range(n.GetChildCount()):walk(n.GetChild(i))
walk(s.GetRootNode());assert counts['skeleton']==52,counts;assert counts['mesh']==1,counts
with np.load(P/'app/flowhmr/output/example/flowhmr_latest.npz') as a,np.load(f.with_suffix('.npz')) as b:
    assert np.array_equal(a['poses'][:,:22],b['poses'][:,:22]);assert np.array_equal(a['trans'],b['trans']);assert np.array_equal(a['betas'],b['betas'])
r={'sdk_reimport':True,**counts,'bytes':f.stat().st_size,'frames':151,'body_root_wrist_preserved':True};(P/'logs/merged_fbx_validation.json').write_text(json.dumps(r,indent=2),encoding='utf-8');print(r);m.Destroy()
