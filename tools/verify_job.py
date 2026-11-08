"""Verify one finished job: frames, FBX variants, project tree, merged contacts."""
import json
import sys
from pathlib import Path

import numpy as np

job = Path(sys.argv[1])
print('JOB', job.name)

manifest_path = job / 'run_manifest.json'
manifest = json.loads(manifest_path.read_text(encoding='utf-8')) if manifest_path.is_file() else {}
frames = manifest.get('frames')
print('  manifest status', manifest.get('status'), 'frames', frames,
      'elapsed', round(manifest.get('elapsed_seconds', 0) / 60, 1), 'min',
      'peak', manifest.get('peak_alloc_GiB'), 'GiB')

with np.load(job / 'body_postprocess.npz') as data:
    body_frames = int(data['num_frames'])
print('  body frames', body_frames, 'ok' if frames == body_frames else 'MISMATCH')

with np.load(job / 'final.npz') as data:
    print('  final.npz poses', data['poses'].shape, 'trans', data['trans'].shape)

for name in ('final.fbx', 'smplh.fbx', 'smplh.npz', 'final.npz', 'validation.json',
             'segments.json', 'correction.json', 'contacts.npz', 'working_30fps.mp4',
             'tracks.npz', 'hamer.log'):
    path = job / name
    print(f'  {name:22} {"OK" if path.is_file() else "MISSING"}',
          f'{path.stat().st_size/1e6:.2f} MB' if path.is_file() else '')

segments = json.loads((job / 'segments.json').read_text(encoding='utf-8')) if (job / 'segments.json').is_file() else {}
rows = segments.get('segments') or []
print('  shots', len(rows), [(row['start'], row['end']) for row in rows])
covered = sum(row['end'] - row['start'] for row in rows)
print('  coverage', covered, '/', segments.get('total_frames'))

correction = json.loads((job / 'correction.json').read_text(encoding='utf-8')) if (job / 'correction.json').is_file() else {}
print('  correction frames', correction.get('frames'), 'segments', len(correction.get('segments') or []),
      'floor', correction.get('floor'))
with np.load(job / 'contacts.npz') as data:
    print('  contacts segments', data['segments'].shape, 'anchors', data['anchor_trajectory'].shape)

validation = json.loads((job / 'validation.json').read_text(encoding='utf-8')) if (job / 'validation.json').is_file() else {}
print('  validation problems', validation.get('problems'))
project = job / 'project' / 'project.json'
print('  project tree', 'OK' if project.is_file() else 'MISSING')
take = next((job / 'project').rglob('takes/*'), None)
if take:
    files = sorted(p.name for p in take.iterdir() if p.is_file())
    print('  take files', len(files), 'fbx', [n for n in files if n.endswith('.fbx')])
