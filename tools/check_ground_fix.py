"""Check the robust ground alignment against real jobs.

For each job it rebuilds the postprocess input from body_origin.npz, runs the
pipeline twice (old "global minimum" alignment vs the new median alignment) and
reports the resulting floor placement.

    envs\\python.exe tools/check_ground_fix.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

PACK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACK / 'app' / 'flowhmr'))

from flowhmr.core.bodymodels.smpl_skeleton import SMPLSkeleton, SMPLMesh  # noqa: E402
from flowhmr.core.math.geometry import (angle_axis_to_rotation_matrix,  # noqa: E402
                                        rotation_matrix_to_rot6d)
from flowhmr.utils.postprocess_utils import _forward_smpl_vertices  # noqa: E402
from flowhmr.utils.runtime_v2m_postprocess_simple import PostprocessPipeline  # noqa: E402

MODEL = PACK / 'app' / 'flowhmr' / 'assets' / 'body_models' / 'smplh' / 'neutral' / 'model.npz'
OUT = PACK / 'outputs' / 'video_to_fbx'

JOBS = [
    ('挑战全网最像模仿杰米超级爆爆回_第二期_1_e695abdc60fb', 'windmill (no foot contacts)'),
    ('example_7fa462d81300', 'ordinary short clip (has foot contacts)'),
]


def build_output(job: Path):
    with np.load(job / 'body_origin.npz') as z:
        poses = torch.from_numpy(z['poses']).float()
        trans = torch.from_numpy(z['trans']).float()
        betas = torch.from_numpy(z['betas']).float().reshape(1, 1, -1)
        contacts = torch.from_numpy(z['foot_contacts']).float()
    T, J, _ = poses.shape
    R = angle_axis_to_rotation_matrix(poses.reshape(-1, 3)).reshape(T, J, 3, 3)
    rot6d = rotation_matrix_to_rot6d(R)
    return {
        'rot6d': rot6d[None].clone(),
        'trans': trans[None].clone(),
        'shapes': betas,
        'foot_contacts': contacts[None].clone(),
        'local_joints_positions': torch.zeros(1, T, 52, 3),
        'pelvis_world': torch.zeros(1, T, 3),
    }


def old_align(self, output):
    v = _forward_smpl_vertices(self.smpl_mesh, output)
    y = v[0, ..., 1].min(dim=-1).values.min().item()
    output['trans'][..., 1] -= y
    output['local_joints_positions'][..., 0, 1] -= y
    output['pelvis_world'][..., 1] -= y
    return output


def per_frame_min(mesh, out):
    with torch.no_grad():
        v = _forward_smpl_vertices(mesh, out)[0]
    return v[..., 1].min(dim=-1).values.numpy()


def report(job: Path, label: str):
    print(f'\n{"="*74}')
    print(f'{label}\njob: {job.name}')
    body = SMPLSkeleton(model_path=str(MODEL)).eval()
    mesh = SMPLMesh(model_path=str(MODEL)).eval()
    pipe = PostprocessPipeline.default(body_model=body, smpl_mesh=mesh)

    base = build_output(job)
    raw = per_frame_min(mesh, base)
    print(f'  raw network output: per-frame lowest point  median={np.median(raw):+.4f} m  min={raw.min():+.4f} m')

    rows = []
    for name, patched in (('old (global min)', True), ('new (median)', False)):
        out = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in base.items()}
        if patched:
            pipe.move_mesh_to_ground = lambda o, m=mesh: old_align(
                type('S', (), {'smpl_mesh': m})(), o)
        else:
            del pipe.move_mesh_to_ground
            pipe.move_mesh_to_ground = PostprocessPipeline.move_mesh_to_ground.__get__(pipe)
        with torch.no_grad():
            res = pipe.run(out)
        shift = float((res['trans'][0, :, 1] - base['trans'][0, :, 1]).median())
        pmin = per_frame_min(mesh, res)
        rows.append((name, shift, float(np.median(pmin)), float(pmin.min()), float(pmin.max())))
        print(f'  {name:16} shift applied={shift:+.4f} m | after: median lowest={np.median(pmin):+.4f} m'
              f'  min={pmin.min():+.4f}  max={pmin.max():+.4f}')

    old_m = rows[0][2]
    new_m = rows[1][2]
    print(f'  => float of ordinary frames: {old_m:+.4f} m  ->  {new_m:+.4f} m')
    return rows


def main():
    for name, label in JOBS:
        job = OUT / name
        if job.is_dir():
            report(job, label)
        else:
            print(f'\nskipped, not found: {job}')


if __name__ == '__main__':
    main()
