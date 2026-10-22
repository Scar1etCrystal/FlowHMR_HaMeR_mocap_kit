"""Report what VGGT concluded about the camera: position, pitch, roll, focal length.

VGGT has no metric scale and no absolute world, so the camera *height* is only
meaningful once the body fixes the floor. This file reports the scale-free parts
(pitch, roll, FOV) and, when final.npz exists, the height above the body's feet.

    envs\\python.exe tools/check_camera_pose.py <job_dir>
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

PACK = Path(__file__).resolve().parents[1]


def decompose(rt_yup: np.ndarray):
    """rt_yup is cam <- y_up_world for one frame."""
    R = rt_yup[:3, :3].astype(np.float64)
    t = rt_yup[:3, 3].astype(np.float64)
    C = -R.T @ t                      # camera centre in the y-up world
    fwd = R.T @ np.array([0.0, 0.0, 1.0])   # OpenCV cameras look down +Z
    up = R.T @ np.array([0.0, -1.0, 0.0])   # OpenCV y points down
    right = R.T @ np.array([1.0, 0.0, 0.0])
    # pitch: angle of the view direction off the horizontal plane
    pitch = math.degrees(math.asin(max(-1.0, min(1.0, fwd[1]))))
    # roll: how far the camera's up-vector is from the world up
    roll = math.degrees(math.acos(max(-1.0, min(1.0, float(up @ np.array([0.0, 1.0, 0.0]))))))
    roll *= -1.0 if right[1] < 0 else 1.0
    return C, fwd, pitch, roll


def main():
    job = Path(sys.argv[1])
    d = np.load(job / 'camera.npz')
    rt = d['camera_RT_yup'] if 'camera_RT_yup' in d.files else d['camera_RT']
    K = d['K']
    hw = d['image_size_hw']
    H, W = int(hw[0]), int(hw[1])
    sampled = int(d['camera_sample_indices'].size) if 'camera_sample_indices' in d.files else -1
    interval = int(d['camera_sampling_interval']) if 'camera_sampling_interval' in d.files else -1

    print(f'job      : {job.name}')
    print(f'frames   : {rt.shape[0]}   sampled={sampled}  interval={interval}')
    print()

    rows = []
    for i in range(rt.shape[0]):
        C, fwd, pitch, roll = decompose(rt[i])
        rows.append((C, fwd, pitch, roll))
    C0, fwd0, pitch0, roll0 = rows[0]
    var = max(abs(r[2] - pitch0) for r in rows)
    print(f'camera centre (y-up world, arbitrary origin): {np.round(C0, 4).tolist()}')
    print(f'  repeats the same pose every frame: pitch spread {var:.6f} deg')
    print()
    print(f'PITCH {pitch0:+.2f} deg   (positive = looking up, negative = looking down, 0 = level)')
    print(f'ROLL  {roll0:+.2f} deg')
    print()
    fx, fy = float(K[0, 0, 0]), float(K[0, 1, 1])
    cx, cy = float(K[0, 0, 2]), float(K[0, 1, 2])
    hfov = math.degrees(2 * math.atan(W / (2 * fx)))
    vfov = math.degrees(2 * math.atan(H / (2 * fy)))
    print(f'intrinsics: fx={fx:.1f} fy={fy:.1f} cx={cx:.1f} cy={cy:.1f}   image {W}x{H}')
    print(f'FOV       : horizontal {hfov:.1f} deg   vertical {vfov:.1f} deg')
    print(f'focal     : {fx / W * 36:.1f} mm equivalent (35mm full frame)')
    print(f'principal point offset from centre: {cx - W / 2:+.1f} px, {cy - H / 2:+.1f} px')
    print()

    # --- height above the floor, once the body defines where the floor is
    fin = job / 'final.npz'
    if fin.is_file():
        sys.path.insert(0, str(PACK / 'app' / 'flowhmr'))
        import torch
        from flowhmr.core.bodymodels.smpl_skeleton import SMPLMesh
        from flowhmr.core.math.geometry import (angle_axis_to_rotation_matrix,
                                                rotation_matrix_to_rot6d)
        from flowhmr.utils.postprocess_utils import _forward_smpl_vertices
        model = PACK / 'app/flowhmr/assets/body_models/smplh/neutral/model.npz'
        with np.load(fin) as z:
            poses = torch.from_numpy(z['poses']).float()
            trans = torch.from_numpy(z['trans']).float()
            betas = torch.from_numpy(z['betas']).float().reshape(1, 1, -1)
        T, J, _ = poses.shape
        R = angle_axis_to_rotation_matrix(poses.reshape(-1, 3)).reshape(T, J, 3, 3)
        out = {'rot6d': rotation_matrix_to_rot6d(R)[None], 'trans': trans[None], 'shapes': betas}
        mesh = SMPLMesh(model_path=str(model)).eval()
        with torch.no_grad():
            v = _forward_smpl_vertices(mesh, out)[0][..., 1].numpy()
        floor = float(v.min(axis=1).min())
        head = float(v.mean(axis=1).max())
        print('body in the same frame (final.npz):')
        print(f'  lowest vertex {floor:+.4f}   highest vertex {head:+.4f}   height {head - floor:.3f}')
        print(f'  camera height above the floor: {C0[1] - floor:+.3f}')
    else:
        print('body (final.npz) not ready yet: camera height needs the floor from the body')

    info = job / 'camera_info.json'
    if info.is_file():
        print()
        print('camera_info.json:', json.dumps(json.loads(info.read_text(encoding='utf-8')), ensure_ascii=False))


if __name__ == '__main__':
    main()
