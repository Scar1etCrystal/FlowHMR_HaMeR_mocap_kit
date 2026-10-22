"""Compare VGGT's camera on a static-camera clip: one frame vs many samples.

The concern: with a fixed camera and a blank background there are no reference
points, so multi-view VGGT has nothing to triangulate and may invent drift or a
spurious tilt. One frame cannot drift, but it also cannot know the true tilt.

    envs\\python.exe tools/compare_camera_sampling.py <job_dir>
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

PACK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACK / 'app' / 'flowhmr'))


def angles(rt_yup: np.ndarray):
    R = rt_yup[:3, :3].astype(np.float64)
    t = rt_yup[:3, 3].astype(np.float64)
    C = -R.T @ t
    fwd = R.T @ np.array([0.0, 0.0, 1.0])
    up = R.T @ np.array([0.0, -1.0, 0.0])
    pitch = math.degrees(math.asin(max(-1.0, min(1.0, fwd[1]))))
    roll = math.degrees(math.acos(max(-1.0, min(1.0, float(up @ np.array([0.0, 1.0, 0.0]))))))
    return C, pitch, roll


def summarise(name: str, path: Path):
    d = np.load(path)
    rt = d['camera_RT_yup'] if 'camera_RT_yup' in d.files else d['camera_RT']
    K = d['K']
    hw = d['image_size_hw']
    H, W = int(hw[0]), int(hw[1])
    n_sampled = int(d['camera_sample_indices'].size) if 'camera_sample_indices' in d.files else -1
    interval = int(d['camera_sampling_interval']) if 'camera_sampling_interval' in d.files else -1
    idx = d['camera_sample_indices'] if 'camera_sample_indices' in d.files else None

    pitches, rolls, centres = [], [], []
    # the per-sample poses are what VGGT actually predicted; the rest is interpolation
    frames = idx if idx is not None and idx.size else np.arange(rt.shape[0])
    for i in frames:
        C, p, r = angles(rt[int(i)])
        pitches.append(p)
        rolls.append(r)
        centres.append(C)
    pitches = np.array(pitches)
    rolls = np.array(rolls)
    centres = np.array(centres)
    fx = float(K[0, 0, 0])
    hfov = math.degrees(2 * math.atan(W / (2 * fx)))

    print(f'\n--- {name}')
    print(f'  sampled frames          : {n_sampled if n_sampled > 0 else rt.shape[0]}   interval={interval}')
    print(f'  pitch  mean {pitches.mean():+7.2f} deg   spread {pitches.max()-pitches.min():6.2f} deg'
          f'   range [{pitches.min():+.2f}, {pitches.max():+.2f}]')
    print(f'  roll   mean {rolls.mean():+7.2f} deg   spread {rolls.max()-rolls.min():6.2f} deg')
    if len(centres) > 1:
        drift = np.linalg.norm(centres - centres[0], axis=1)
        step = np.linalg.norm(np.diff(centres, axis=0), axis=1)
        print(f'  camera drift over the clip: max {drift.max():.4f}  (per-sample step max {step.max():.4f})')
    else:
        print('  camera drift over the clip: 0 by construction (one pose repeated)')
    print(f'  hFOV {hfov:.2f} deg   fx {fx:.1f}   ({fx / W * 36:.1f} mm equiv)')
    return pitches, rolls, fx, hfov


def main():
    job = Path(sys.argv[1])
    single = job / 'camera.npz'
    multi = job / 'camera_multiframe.npz'
    if not single.is_file():
        raise SystemExit(f'no camera.npz in {job}')
    if not multi.is_file():
        raise SystemExit(f'no camera_multiframe.npz in {job} - run it first')
    summarise('single frame (what the new mode produced)', single)
    summarise('multi frame (the default auto/31 sampling)', multi)
    print('\ninterpretation:')
    print('  the single-frame camera is the identity by construction - no tilt to get wrong,')
    print('  but also no way to discover a real tilt. Whether multi-view invents one is what')
    print('  the spread above shows.')


if __name__ == '__main__':
    main()
