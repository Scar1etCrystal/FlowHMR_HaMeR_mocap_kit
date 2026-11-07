"""Compare t1499_H4 runs: HaMeR N=1/raw hands vs the accepted default.

Baseline A : sam_hand_mode=mean (N60) + HaMeR interval=2   [existing job]
Run B      : sam_hand_mode=mean (N60) + HaMeR interval=1
Run C      : sam_hand_mode=per_frame + HaMeR interval=1

A vs B isolates the HaMeR sampling; B vs C isolates the SAM hand condition.
All three go through the same no-smooth postprocess and the same merge.
"""
from pathlib import Path
import json
import numpy as np
from scipy.spatial.transform import Rotation

P = Path(r'F:\flowhmr_pack')
JOB_ROOT = P/'outputs/video_to_fbx'
VIDEO_NAME = 't1499_H4'
OUT = P/'outputs/video_to_fbx_comparison'
OUT.mkdir(parents=True, exist_ok=True)
FINGER = list(range(22, 52))
BODY = list(range(0, 22))


def discover():
    """Find completed jobs for this clip and classify them by their own settings."""
    found = {}
    for job in sorted(JOB_ROOT.glob(f'{VIDEO_NAME}_*')):
        req, stg = job/'request.json', job/'settings.json'
        if not (req.is_file() and stg.is_file() and (job/'final.npz').is_file()):
            continue
        if VIDEO_NAME not in json.loads(req.read_text(encoding='utf-8')).get('source', ''):
            continue
        c = json.loads(stg.read_text(encoding='utf-8'))
        mode = c.get('sam_hand_mode', 'mean')
        ham = c.get('hamer_interval', 2)
        label = f"{'perframe' if mode == 'per_frame' else 'samN' + str(c.get('sam_hand_interval'))}" \
                f"_hamerN{ham}"
        found[label] = job
    return found


def geo(a, b, joints):
    ra = Rotation.from_rotvec(a[:, joints].reshape(-1, 3))
    rb = Rotation.from_rotvec(b[:, joints].reshape(-1, 3))
    d = np.degrees((ra.inv()*rb).magnitude()).reshape(len(a), len(joints))
    return {'mean': float(d.mean()), 'p99': float(np.percentile(d, 99)), 'max': float(d.max())}


def load(job):
    with np.load(job/'final.npz', allow_pickle=False) as z:
        return {k: z[k].copy() for k in z.files}


def main():
    jobs = discover()
    print('discovered jobs:')
    for k, v in jobs.items():
        print(f'  {k}  ->  {v.name}')
    data = {}
    for name, job in jobs.items():
        data[name] = load(job)
        info = json.loads((job/'generation_info.json').read_text(encoding='utf-8'))
        print(f"{name}: frames={len(data[name]['poses'])} contacts={info.get('foot_contacts_exported')}")
    report = {'jobs': {k: str(v) for k, v in jobs.items()}, 'runs_present': sorted(data)}
    if len(data) < 2:
        print('not enough completed runs to compare')
        (OUT/'comparison.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        return 0

    def pair(a, b):
        A, B = data[a], data[b]
        assert len(A['poses']) == len(B['poses']), (a, b)
        tr = np.linalg.norm(A['trans']-B['trans'], axis=1)*100
        ra = Rotation.from_rotvec(A['poses'].reshape(-1, 3))
        rb = Rotation.from_rotvec(B['poses'].reshape(-1, 3))
        per_joint = np.degrees((ra.inv()*rb).magnitude()).reshape(len(A['poses']), 52).mean(axis=0)
        return {'body_joints_0_21_deg': geo(A['poses'], B['poses'], BODY),
                'finger_joints_22_51_deg': geo(A['poses'], B['poses'], FINGER),
                'root_translation_cm': {'mean': float(tr.mean()), 'max': float(tr.max())},
                'mean_deg_per_body_joint': [round(float(v), 4) for v in per_joint[:22]],
                'mean_deg_per_finger_joint': [round(float(v), 4) for v in per_joint[22:]],
                'equal_foot_contacts': bool(np.array_equal(A['foot_contacts'], B['foot_contacts']))}

    names = sorted(data)
    comparisons = {}
    for i, a in enumerate(names):
        for b in names[i+1:]:
            comparisons[f'{a}  vs  {b}'] = pair(a, b)
    report['comparisons'] = comparisons

    # Contact statistics per run: a body-level sanity signal independent of the hands.
    stats = {}
    for name, d in data.items():
        fc = d['foot_contacts']
        stats[name] = {'frac_gt_0.7_per_column': (fc > 0.7).mean(axis=0).round(4).tolist(),
                       'columns': ['L_Ankle', 'L_Foot', 'R_Ankle', 'R_Foot']}
    report['foot_contact_stats'] = stats
    (OUT/'comparison.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(comparisons, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
