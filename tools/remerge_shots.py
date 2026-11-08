"""Re-merge a finished multi-shot job from its per-shot npz files.

Useful after the merge logic changes (for example heading continuity): the
per-shot inference results are reused, so no GPU work is repeated. Only the
body arrays, the Humanoid52/SMPL-H exports, validation and project tree are
rebuilt.

Usage: python tools/remerge_shots.py <job_dir>
"""
import json
import sys
from pathlib import Path

PACK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACK / 'tools'))

import numpy as np  # noqa: E402


def main() -> int:
    job = Path(sys.argv[1]).resolve()
    sys.path.insert(0, str(PACK / 'tools'))
    import multishot
    import video_to_fbx

    cfg = video_to_fbx.load_settings(job / 'settings.json')
    video_to_fbx.configure(cfg)

    plan = json.loads((job / 'segments.json').read_text(encoding='utf-8'))
    seed = int(cfg['seed'])
    parts = []
    for index, segment in enumerate(plan['segments']):
        npz = job / 'segments' / 'official' / f'shot{index:02d}' / f'shot{index:02d}_seed{seed}.npz'
        if not npz.is_file():
            raise SystemExit(f'missing segment result: {npz}')
        with np.load(npz) as data:
            parts.append({key: data[key] for key in data.files})
        print(f'  shot{index:02d}: {len(parts[-1]["poses"])} frames', flush=True)

    merged = multishot.merge_segment_arrays(parts)
    np.savez(job / 'body_postprocess.npz', **merged)
    print(f'  merged {merged["num_frames"]} frames -> body_postprocess.npz', flush=True)

    video_to_fbx.run_merge(job)
    print('REMERGED', job, flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
