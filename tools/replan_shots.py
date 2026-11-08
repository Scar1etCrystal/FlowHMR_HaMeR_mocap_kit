"""Re-plan shot segments from an existing tracker npz (no detector run)."""
import json
import sys
from pathlib import Path

PACK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACK / 'tools'))

import multishot  # noqa: E402

NPZ = Path(sys.argv[1])
PLAN = Path(sys.argv[2]) if len(sys.argv) > 2 else NPZ.with_name(NPZ.stem + '_plan.json')

data = multishot.load_tracks(NPZ)
segments = multishot.plan_segments(data['tracks'], data['total_frames'], data['width'], data['height'])
plan = multishot.describe_plan(data['tracks'], data['total_frames'], data['width'], data['height'], segments)
PLAN.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding='utf-8')
print('TOTAL', data['total_frames'], 'SHOTS', len(segments))
for segment in segments:
    print('  {:>5}-{:>5}  frames {:>5}  ids {}  frac {:.3f}'.format(
        segment['start'], segment['end'], segment['frames'], segment['track_ids'], segment['bbox_fraction']))
covered = sum(segment['frames'] for segment in segments)
print('COVERED', covered, '/', data['total_frames'])
