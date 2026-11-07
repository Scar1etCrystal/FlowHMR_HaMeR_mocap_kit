from pathlib import Path
import json,pickle,numpy as np
from scipy.spatial.transform import Rotation
P=Path(__file__).resolve().parents[1]
rows=[];infos={}
def load(n):
    d=P/f'outputs/hamer_pose_n{n}';infos[n]=json.loads((d/'run_info.json').read_text())
    with (d/'example_30fps_hamer_data.pkl').open('rb') as f:z=pickle.load(f)
    with (d/'sampled_raw.pkl').open('rb') as f:raw=pickle.load(f)
    # Sampling invariant: BOTH models run on selected frames only.
    assert all(x['frame_idx']%n==0 for x in raw['observations'])
    assert all(x%n==0 for x in raw['sampled_frames'])
    return z,raw,np.load(d/'example_30fps_sampling.npz')['source']
def stats(a,b):
    d=np.degrees((Rotation.from_matrix(a.reshape(-1,3,3)).inv()*Rotation.from_matrix(b.reshape(-1,3,3))).magnitude())
    return {'mean_deg':float(d.mean()),'p99_deg':float(np.percentile(d,99)),'max_deg':float(d.max())}
base,raw1,src1=load(1)
for n in (2,3):
    z,raw,src=load(n);valid=base['global_orient_valid'].astype(bool)&z['global_orient_valid'].astype(bool)
    row={'N':n,'frames':len(z['hand_pose']),'valid_LR':valid.sum(axis=0).tolist(),'hand_all':stats(base['hand_pose'][valid],z['hand_pose'][valid]),'wrist_all':stats(base['global_orient'][valid],z['global_orient'][valid])}
    mask=valid&(src==2)
    row['hand_interpolated_only']=stats(base['hand_pose'][mask],z['hand_pose'][mask])
    row['inference_speedup']=infos[1]['inference_seconds']/infos[n]['inference_seconds'];row['total_speedup']=infos[1]['seconds_with_load_export']/infos[n]['seconds_with_load_export']
    # Compare raw observations at the selected frames (no smoothing lookahead).
    rb={(x['frame_idx'],x['is_right']):x['results']['pose'] for x in raw1['observations']}
    pairs=[(rb[(x['frame_idx'],x['is_right'])],x['results']['pose']) for x in raw['observations'] if (x['frame_idx'],x['is_right']) in rb]
    row['observed_raw_consistency']=stats(np.stack([x[0] for x in pairs]),np.stack([x[1] for x in pairs]))
    rows.append(row)
report={'run_info':infos,'rows':rows,'scope':'same existing HaMeR/ViTPose configuration; raw sparse inference THEN interpolate THEN original smoothing; this is real inference, not sampling a smoothed baseline'}
(P/'logs/hamer_pose_sampling_validation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
s=['# 真正隔帧推理验证（替代之前离线抽样的速度/精度估计）','',report['scope'],'','- 所有推理任务严格满足 f % N == 0。','- 两种模式均 TTA=1、batch=16、完整序列平滑窗口=11。','- 旋转SLERP，betas/cam_t线性；有效性和插值来源独立。','', '| N | ViTPose调用 | HaMeR手裁剪 | 推理耗时 | 含加载/导出的总耗时 |','|---|---:|---:|---:|---:|']
for n in (1,2,3):
    i=infos[n];s.append(f"| {n} | {i['vitpose_calls']} | {i['hamer_hand_calls']} | {i['inference_seconds']:.2f}s | {i['seconds_with_load_export']:.2f}s |")
s+=['','| N | 手指旋转平均差 | p99 | 最大 | 只看插值帧平均差 | 手腕平均差 |','|---|---:|---:|---:|---:|---:|']
for r in rows:
    d=r['hand_all'];s.append(f"| {r['N']} | {d['mean_deg']:.4f}° | {d['p99_deg']:.4f}° | {d['max_deg']:.4f}° | {r['hand_interpolated_only']['mean_deg']:.4f}° | {r['wrist_all']['mean_deg']:.4f}° |")
s+=['','误差相对于同配置的逐帧HaMeR输出，不是真实动作标注。帧数151、30fps、单片段，不代表所有手速。','旧报告0.747°/1.527°是对已经平滑后的逐帧姿态抽样，不等于真实稀疏推理结果；68.2秒还是含额外诊断的总时间，不能再用68/N估计加速。','当前ViTPose配置载入权重存在unexpected expert keys；N1/N2的相对对照可比，但绝对关键点质量仍需另审。']
(P/'logs/HAMER_POSE_SAMPLING_VALIDATION.md').write_text('\n'.join(s)+'\n',encoding='utf-8')
print(json.dumps(report,indent=2))
