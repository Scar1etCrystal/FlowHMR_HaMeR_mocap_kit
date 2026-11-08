"""Wrap a finished job as a toyxyz_RGBMocap project tree.

The application has no "import motion" entry point: it opens its own project
layout and only then retargets. This module writes that layout around the
Humanoid52 export so the same take can be opened (or retargeted headlessly)
inside the original application.

Layout produced under ``<job>/project``:

    project.json                        workspace_version 2, active_model flowhmr
    media/<video>
    flowhmr/project.json                workspace stage record
    flowhmr/people.json
    flowhmr/people_scene.json
    flowhmr/input.mp4
    flowhmr/people/<scene>/person_0001/project.json
    flowhmr/people/<scene>/person_0001/tracking.npz
    flowhmr/people/<scene>/person_0001/takes/<first>_<last>/   <- export lives here
"""
from __future__ import annotations

import hashlib
import json
import shutil
import time
import uuid
from pathlib import Path

import numpy as np

KIND = 'corrected_initial'


def _stat(path: Path):
    value = Path(path).stat()
    return [int(value.st_size), int(value.st_mtime_ns)]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _probe_video(path: Path) -> dict:
    import cv2
    cap = cv2.VideoCapture(str(path))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return dict(path=str(path), width=width, height=height, fps=fps,
                duration=frames / fps if fps else 0.0, frames=frames,
                codec='h264', bytes=int(Path(path).stat().st_size))


def prepare(job: Path, video: Path, *, frames: int, fps: float = 30.0,
            name: str | None = None, reset: bool = False) -> dict:
    """Create the project skeleton and return the paths the export must fill."""
    job = Path(job)
    video = Path(video)
    info = _probe_video(video) if video.is_file() else dict(
        path=str(video), width=0, height=0, fps=fps, duration=frames / fps,
        frames=frames, codec='h264', bytes=0)
    root = job / 'project'
    if root.exists() and reset:
        shutil.rmtree(root)
    existing = sorted((root / 'flowhmr' / 'people').glob('*/person_*/takes/*')) if (root / 'flowhmr' / 'people').is_dir() else []
    if existing:
        take = existing[0]
        person = take.parents[1]
        scene = person.parent.name
        take.mkdir(parents=True, exist_ok=True)
    else:
        scene = uuid.uuid4().hex
        person = root / 'flowhmr' / 'people' / scene / 'person_0001'
        take = person / 'takes' / f'{0:09d}_{int(frames):09d}'
        take.mkdir(parents=True)
    media = root / 'media'
    media.mkdir(parents=True, exist_ok=True)
    target_video = media / video.name
    if video.is_file() and not target_video.exists():
        shutil.copyfile(video, target_video)
    for destination in (root / 'flowhmr' / 'input.mp4', person / 'input.mp4'):
        if video.is_file() and not destination.exists():
            shutil.copyfile(video, destination)
    np.savez(person / 'tracking.npz', boxes=np.zeros((frames, 4), np.float32))
    return dict(root=root, take=take, person=person, scene=scene, video=video,
                media_video=target_video, info=info, frames=int(frames), fps=float(fps),
                name=name or job.name)


def finalize(paths: dict) -> dict:
    """Write project.json / people.json around the already exported take."""
    root = paths['root']
    take = paths['take']
    person = paths['person']
    scene = paths['scene']
    frames = paths['frames']
    fps = paths['fps']
    info = paths['info']
    take_rel = take.relative_to(root).as_posix()
    person_rel = person.relative_to(root).as_posix()
    media_rel = paths['media_video'].relative_to(root).as_posix() if paths['media_video'].is_file() else None
    step = 1.0 / fps
    times = [round(index * step, 12) for index in range(frames)]
    source = dict(info)
    source['fps'] = fps
    source['frames'] = frames
    source['duration'] = frames / fps
    source_sha = _sha256(paths['media_video']) if paths['media_video'].is_file() else None
    stages = {'flow_camera': {'status': 'complete', 'elapsed': 0.0,
                              'output': str((root / 'flowhmr' / 'flow_camera.npz'))}}
    flow_options = {'camera_interval': 0, 'fast_sam': True, 'sam_batch4': False,
                    'sam_gpu_crop': True, 'sam_hand_stride4': False, 'hamer': True}
    correction = {'auto_floor': True, 'floor': 0.0, 'preset': 'normal',
                  'pelvis_offset': 0.0, 'mode': 'flat', 'foot_lock': True,
                  'hand_lock': False}

    def base_project(workspace_root: str, result: str) -> dict:
        return {
            'version': 1, 'source': source, 'source_sha256': source_sha,
            'start': 0.0, 'end': frames / fps, 'fps': fps, 'frames': frames,
            'width': info.get('width', 0), 'height': info.get('height', 0),
            'camera': 'static', 'stages': stages,
            'pipeline': {'model': 'flowhmr', 'result': result,
                         'htd_refine': False, 'footmr': False},
            'timeline': {'start': 0.0, 'step': step, 'frames': frames},
            'workspace_root': workspace_root, 'original_times': times,
            'flow_options': flow_options,
        }

    take_project = base_project('..\\..\\..\\..\\..\\..', KIND)
    take_project.update(
        people_root='..\\..\\..\\..\\..', person_id=1, person_preview=False,
        person_frames=[0, frames],
        person_fingerprint=_sha256(paths['media_video']) if paths['media_video'].is_file() else None,
        person_media_version=1, correction=correction,
        exmr_options={'static_camera': True, 'smoothing': 'global'},
        prompthmr_options={'people': 1, 'interaction': False})
    (take / 'project.json').write_text(json.dumps(take_project, indent=2), encoding='utf-8')

    person_project = base_project('..\\..\\..\\..', KIND)
    person_project.update(person_id=1, person_frames=[0, frames], correction=correction)
    (person / 'project.json').write_text(json.dumps(person_project, indent=2), encoding='utf-8')

    workspace = base_project('..', 'corrected_initial')
    (root / 'flowhmr' / 'project.json').write_text(json.dumps(workspace, indent=2), encoding='utf-8')

    workspace_input = root / 'flowhmr' / 'input.mp4'
    people = {
        'version': 1,
        'input': _stat(workspace_input) if workspace_input.is_file() else None,
        'fingerprint': source_sha,
        'people': [{
            'id': 1, 'name': 'Person 1', 'color': '#969696', 'checked': True,
            'first': 0, 'last': frames, 'observations': frames,
            'preview': person_rel,
            'takes': [{'project': take_rel, 'first': 0, 'end': frames}],
            'observation_id': 1,
        }],
    }
    (root / 'flowhmr' / 'people.json').write_text(json.dumps(people, indent=2), encoding='utf-8')
    media_stat = _stat(workspace_input) if workspace_input.is_file() else None
    people_scene = {
        'version': 2, 'input': media_stat,
        'anchor': take_rel, 'anchor_id': 1,
        'world_to_camera': [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0],
                            [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
        'sources': {path.name: _stat(path) for path in take.glob('*') if path.is_file()},
    }
    (root / 'flowhmr' / 'people_scene.json').write_text(
        json.dumps(people_scene, indent=2), encoding='utf-8')
    (root / 'flowhmr' / 'scene_floor.json').write_text(json.dumps(dict(
        version=1, auto_floor=True, floor=0.0), indent=2), encoding='utf-8')

    root_project = {
        'workspace_version': 2,
        'name': paths['name'],
        'source': source,
        'media': media_rel,
        'time_offset': 0.0,
        'active_model': 'flowhmr',
        'models': {'flowhmr': {'active': take_rel,
                               'view': {'camera_background': False,
                                        'source_time': min(1.0, frames * step),
                                        'trim_time': [0.0, frames / fps]}}},
        'origins': [{'path': str(root), 'relative': '.'}],
        'reference': take_rel,
    }
    (root / 'project.json').write_text(json.dumps(root_project, indent=2), encoding='utf-8')
    return dict(root=str(root), take=str(take), project=str(root / 'project.json'),
                people=str(root / 'flowhmr' / 'people.json'), frames=frames, fps=fps)


def build(job: Path, video: Path, *, frames: int, fps: float = 30.0,
          exporter=None, emit=print) -> dict:
    """Prepare the skeleton, run the exporter into the take, then finalize."""
    paths = prepare(job, video, frames=frames, fps=fps)
    if exporter is not None:
        exporter(paths['take'])
    return finalize(paths)
