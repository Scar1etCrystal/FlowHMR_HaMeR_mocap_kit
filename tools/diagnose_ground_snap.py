"""Diagnose the ground-snapping step on one finished job.

Reconstructs the SMPL-H mesh for body_origin (raw network output) and
body_postprocess, then reports the per-frame lowest vertex so we can see what
the rigid "move_mesh_to_ground" shift actually anchored to.

    envs\\python.exe tools/diagnose_ground_snap.py <job_dir>
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

PACK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACK / 'app' / 'flowhmr'))

from flowhmr.core.bodymodels.smpl_skeleton import SMPLSkeleton, SMPLMesh  # noqa: E402
from flowhmr.core.math.geometry import (angle_axis_to_rotation_matrix,  # noqa: E402
                                        rotation_matrix_to_rot6d)
from flowhmr.utils.postprocess_utils import (  # noqa: E402
    _forward_smpl_vertices, _forward_smpl_joints)

MODEL = PACK / 'app' / 'flowhmr' / 'assets' / 'body_models' / 'smplh' / 'neutral' / 'model.npz'

JOINTS = {'L_Ankle': 7, 'L_Foot': 10, 'R_Ankle': 8, 'R_Foot': 11,
          'L_Wrist': 20, 'R_Wrist': 21, 'Head': 15, 'Pelvis': 0}


def load_npz(path: Path):
    with np.load(path) as z:
        poses = torch.from_numpy(z['poses']).float()      # (T, 52, 3) axis-angle
        trans = torch.from_numpy(z['trans']).float()      # (T, 3)
        betas = torch.from_numpy(z['betas']).float().reshape(1, 1, -1)
    T, J, _ = poses.shape
    R = angle_axis_to_rotation_matrix(poses.reshape(-1, 3)).reshape(T, J, 3, 3)
    rot6d = rotation_matrix_to_rot6d(R)                   # (T, J, 6)
    return {'rot6d': rot6d[None], 'trans': trans[None], 'shapes': betas}


def main():
    job = Path(sys.argv[1])
    mesh = SMPLMesh(model_path=str(MODEL)).eval()
    body = SMPLSkeleton(model_path=str(MODEL)).eval()

    print(f'job: {job.name}')
    for name in ('body_origin.npz', 'body_postprocess.npz'):
        p = job / name
        if not p.is_file():
            continue
        out = load_npz(p)
        with torch.no_grad():
            verts = _forward_smpl_vertices(mesh, out)[0]          # (T, V, 3)
            joints = _forward_smpl_joints(body, out)[0][0]        # (T, J, 3)
        v = verts[..., 1].min(dim=-1).values.numpy()              # per-frame lowest vertex Y
        T = len(v)
        order = np.argsort(v)
        med = float(np.median(v))
        print(f'\n=== {name}  ({T} frames)')
        print(f'  global lowest vertex Y : {v.min():+.4f} m  at frame {int(order[0])}')
        print(f'  median of per-frame min: {med:+.4f} m')
        print(f'  p05 / p25 / p75        : {np.percentile(v,5):+.4f} / {np.percentile(v,25):+.4f} / {np.percentile(v,75):+.4f}')
        print(f'  highest per-frame min  : {v.max():+.4f} m')
        print(f'  shift applied (by min) : {-v.min():+.4f} m')
        print(f'  shift by median        : {-med:+.4f} m   (diff {(-med)-(-v.min()):+.4f} m)')
        thr = np.percentile(v, 10)
        low = np.where(v <= thr)[0]
        print(f'  lowest-decile frames   : {low[:20].tolist()}{" ..." if low.size>20 else ""}'
              f'  ({low.size}/{T} = {100*low.size/T:.0f}% of the clip)')
        fmin = int(order[0])
        print(f'  at the lowest frame {fmin}:')
        for jn, ji in JOINTS.items():
            print(f'      {jn:8} Y = {joints[fmin, ji, 1]:+.4f}')
        # what the feet are doing in the frames that are lifted
        high = np.where(v >= np.percentile(v, 75))[0]
        foot_y = joints[:, JOINTS['L_Foot'], 1].numpy()
        print(f'  L_Foot height, median over the whole clip: {np.median(foot_y):+.4f} m')
        print(f'  L_Foot height, median over the highest-quartile frames: {np.median(foot_y[high]):+.4f} m')


if __name__ == '__main__':
    main()
