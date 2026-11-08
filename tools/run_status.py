"""Compact progress of the batched video runs (no giant SAM logs)."""
import json
from pathlib import Path

ROOT = Path(r'F:\flowhmr_pack\outputs\video_to_fbx')
JOBS = [
    'neverending_1080p30_f25eb44e22',
    'ribbon_1080p30_135846b63b',
    'kanko_mirrored_1080p30_67cbdd1f21',
]
for name in JOBS:
    job = ROOT / name
    print(f'== {name}')
    if not job.is_dir():
        print('   (not created yet)')
        continue
    plan = job / 'segments.json'
    if plan.is_file():
        shots = json.loads(plan.read_text(encoding='utf-8')).get('segments') or []
        done = 0
        for index, shot in enumerate(shots):
            shot_dir = job / 'segments' / 'official' / f'shot{index:02d}'
            npz = shot_dir / f'shot{index:02d}_seed0.npz'
            marks = []
            if (shot_dir / f'shot{index:02d}_sam3d_feat_static.pt').is_file():
                marks.append('sam')
            if npz.is_file():
                marks.append('npz')
                done += 1
            print(f'   shot{index:02d} {shot["start"]:>5}-{shot["end"]:>5} '
                  f'{shot["frames"]:>5}f  {" ".join(marks) or "queued"}')
        print(f'   shots done: {done}/{len(shots)}')
    else:
        print('   (planning)')
    for marker in ('body_postprocess.npz', 'working_30fps.mp4', 'hands/working_30fps_hamer_data.pkl',
                   'final.npz', 'final.fbx', 'smplh.fbx', 'smplh.npz', 'validation.json',
                   'correction.json', 'contacts.npz'):
        path = job / marker
        if path.is_file():
            print(f'   {marker:46} {path.stat().st_size / 1e6:.2f} MB')
    manifest = job / 'run_manifest.json'
    if manifest.is_file():
        data = json.loads(manifest.read_text(encoding='utf-8'))
        print(f'   DONE status={data.get("status")} frames={data.get("frames")} '
              f'elapsed={data.get("elapsed_seconds", 0) / 60:.1f}min '
              f'peak={data.get("peak_alloc_GiB")}GiB')
