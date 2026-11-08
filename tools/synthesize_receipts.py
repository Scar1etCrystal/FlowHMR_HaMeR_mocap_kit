"""Synthesise the scene-camera and shared-floor receipts the app validates.

The application treats a take as "current" only when the correction report
carries a shared-floor receipt whose signature matches a locally resolved
floor. That receipt embeds machine-local stamps (body model, media), so it has
to be produced on the machine that opens the project - which is what this
script does, using the application's own modules.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np

PACK = Path(__file__).resolve().parents[1]
MODEL_SOURCE = PACK / 'app' / 'flowhmr' / 'assets' / 'body_models' / 'smplh' / 'neutral' / 'model.npz'
IDENTITY = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]


def stamp(path: Path):
    state = Path(path).stat()
    return [int(state.st_size), int(state.st_mtime_ns)]


def camera_k(take: Path) -> list:
    """Use the intrinsics the export recorded, falling back to a 50 mm guess."""
    candidates = sorted(take.parents[6].glob('official/*/*_static_camera.npz'))
    for path in candidates:
        with np.load(path) as data:
            if 'K' in data.files:
                return np.asarray(data['K'], float)[0].tolist()
    return [[1000.0, 0.0, 960.0], [0.0, 1000.0, 540.0], [0.0, 0.0, 1.0]]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('take', help='take directory (contains corrected_initial_*.json)')
    ap.add_argument('--app', default=os.environ.get('TOYXYZ_APP', r'F:\_toyxyz_app'))
    args = ap.parse_args()
    app = Path(args.app).resolve()
    take = Path(args.take).resolve()
    workspace = take.parents[4]           # .../flowhmr
    project = workspace / 'project.json'
    if not project.is_file():
        raise SystemExit(f'not a workspace: {project}')

    model = app / 'models' / 'body_models' / 'smplh' / 'neutral' / 'model.npz'
    if not model.is_file():
        model.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(MODEL_SOURCE, model)
        print(f'  model -> {model}')

    for name in ('input.mp4', 'tracking.npz'):
        source = take.parent / name
        if not source.is_file():
            source = workspace / name
        if source.is_file() and not (take / name).is_file():
            shutil.copyfile(source, take / name)
            print(f'  {name} -> take')

    os.environ['RGB_MOCAP_DATA'] = str(app)
    sys.path.insert(0, str(app))
    sys.path.insert(0, str(PACK / 'app' / 'flowhmr'))   # upstream body model code
    from mocap import scene_camera, scene_floor
    from mocap.flow_floor import resolve as resolve_floor

    # The viewer also wants the camera stage output at the workspace root.
    camera_path = workspace / 'flow_camera.npz'
    if not camera_path.is_file():
        frames = int(json.loads(project.read_text(encoding='utf-8'))['frames'])
        height = int(json.loads(project.read_text(encoding='utf-8'))['height'])
        width = int(json.loads(project.read_text(encoding='utf-8'))['width'])
        k = np.asarray(camera_k(take), float)
        raw = np.tile(np.eye(4), (frames, 1, 1)).astype(np.float32)
        yup = raw.copy()
        yup[:, 1, 1] = -1.0
        yup[:, 2, 2] = -1.0
        np.savez(camera_path, camera_RT=raw, RT=raw, RT_vggt=raw, camera_RT_yup=yup,
                 K=np.repeat(k[None], frames, axis=0),
                 image_size_hw=np.array([height, width]),
                 camera_sampling_interval=-1, camera_sample_indices=np.array([], np.int64),
                 camera_world='explicit_static_opencv', k_space='original_frame',
                 camera_policy_version=1)
        print(f'  flow_camera.npz: {frames} frames')

    sources = scene_camera.camera_sources(take)
    record = dict(version=scene_camera.VERSION, sources=sources, model=stamp(model),
                  world_to_camera=IDENTITY, K=camera_k(take),
                  htd_compatible=False, source_model='flowhmr',
                  method='static camera (kit export)')
    (take / 'scene_camera.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
    print(f'  scene_camera.json: {len(sources)} sources')

    if not (workspace / 'scene_floor.json').is_file():
        (workspace / 'scene_floor.json').write_text(json.dumps(
            dict(version=1, auto_floor=True, floor=0.0), indent=2), encoding='utf-8')
    options = scene_floor.read_config(workspace)
    resolved = resolve_floor(workspace, options)
    print(f'  scene_floor_resolved.json: floor={resolved["floor"]:.6f} '
          f'samples={resolved.get("samples")}')

    shared = dict(options=resolved['options'], inputs=resolved['inputs'],
                  floor=resolved['floor'], signature=resolved['signature'])
    patched = 0
    for name in ('corrected_initial_correction.json', 'correction.json'):
        path = take / name
        if not path.is_file():
            continue
        payload = json.loads(path.read_text(encoding='utf-8'))
        payload['shared_floor'] = shared
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding='utf-8')
        patched += 1
    for name in ('corrected_initial_mixamo.json',):
        path = take / name
        if path.is_file():
            payload = json.loads(path.read_text(encoding='utf-8'))
            if isinstance(payload.get('correction'), dict):
                payload['correction']['shared_floor'] = shared
                path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding='utf-8')
                patched += 1
    print(f'  shared_floor receipt written into {patched} manifest(s)')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
