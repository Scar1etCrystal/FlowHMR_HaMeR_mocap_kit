"""Video -> FBX through the official FlowHMR pipeline plus our HaMeR finger pass.

One process, one pass:
  official FlowHMR (transcode -> tracking -> camera -> SAM tokens -> generation
  -> RGBMocap/Kimodo postprocess) -> HaMeR fingers -> SMPL-H merge -> FBX

The console starts this file as ``python video_to_fbx.py <video> --settings <cfg>``
and follows the run through the same markers the previous staged runner printed:

    Output: <job dir>
    === prepare === / === camera === / === sam === / === generate ===
    === hamer === / === merge ===
    >>> bbox detection: N/M frames
    [sam extract] N/M frames
    GENERATION_DONE <frames>
    DONE: <job>/final.fbx

No VRAM estimation or thresholds here: measured usage changes with the model
versions, so the runner reports what actually happened instead of predicting.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

PACK = Path(__file__).resolve().parents[1]
TOOLS = PACK / 'tools'
APP = PACK / 'app' / 'flowhmr'
PY = sys.executable if sys.executable else str(PACK / 'envs' / 'python.exe')

DEFAULTS = {
    'seed': 0,
    'cfg_scale': 1.0,
    'steps': None,
    'sam_hand_stride': 2,
    'sam_hand_seed': 20261014,
    'sam_chunk_size': 16,
    'static_camera': True,
    'vggt_interval': 10,
    'attention_backend': 'query_batched',
    'attention_query_chunk': 256,
    'video_encoder': 'libx264',
    'hamer_interval': 2,
    'hamer_batch_size': 16,
    'hamer_smooth_window': 11,
    'hamer_max_interp_gap': 15,
    'multi_shot': True,
    'shot_min_frames': 24,
}

STAGE_PRINT = {
    'preprocess': 'prepare',
    'camera': 'camera',
    'feature_extract': 'sam',
    'generation': 'generate',
}


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def save_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')


def sha_text(text: str) -> str:
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


# The console page still sends the previous settings names; keep them working
# instead of forcing a UI rewrite in the same pass.
LEGACY = {
    'sam_hand_interval': 'sam_hand_stride',
    'sam_sampling_seed': 'sam_hand_seed',
    'flow_seed': 'seed',
    'flow_steps': 'steps',
    'flow_cfg_scale': 'cfg_scale',
}
OBSOLETE = ('sam_hand_mode', 'vggt_max_samples', 'max_generation_frames', 'threads')


def load_settings(path: Path) -> dict:
    cfg = dict(DEFAULTS)
    if not path.is_file():
        return cfg
    raw = json.loads(path.read_text(encoding='utf-8'))
    applied, ignored = [], []
    for key, value in raw.items():
        target = LEGACY.get(key, key)
        if target in DEFAULTS:
            if target == 'vggt_interval':
                try:
                    value = int(value)
                except (TypeError, ValueError):
                    value = DEFAULTS['vggt_interval']  # "auto" is gone; keep a fixed stride
            cfg[target] = value
            applied.append(f'{key}->{target}' if key != target else key)
        elif key not in OBSOLETE:
            ignored.append(key)
    print(f'  settings: {", ".join(applied) or "defaults"}', flush=True)
    if ignored:
        print(f'  settings ignored: {", ".join(ignored)}', flush=True)
    return cfg


def job_name(video: Path, cfg: dict) -> str:
    stat = video.stat()
    key = sha_text(json.dumps({
        'video': str(video),
        'size': stat.st_size,
        'mtime': int(stat.st_mtime),
        'settings': {k: cfg[k] for k in sorted(DEFAULTS)},
    }, sort_keys=True))
    stem = ''.join(ch if ch.isalnum() or ch in '-_.' else '_' for ch in video.stem)[:64]
    return f'{stem}_{key[:10]}'


def configure(cfg: dict) -> None:
    """Environment the official code reads before/at import time."""
    os.environ['PYTHONUTF8'] = '1'
    os.environ.setdefault('PYTHONIOENCODING', 'utf-8')
    os.environ.setdefault('MHR_USE_CUDA_GRAPH', '0')
    os.environ.setdefault('USE_COMPILE', '0')
    os.environ.setdefault('PARALLEL_DECODERS', '0')
    os.environ['FLOWHMR_ATTENTION_BACKEND'] = str(cfg['attention_backend'])
    os.environ['FLOWHMR_ATTENTION_QUERY_CHUNK_SIZE'] = str(int(cfg['attention_query_chunk']))
    os.environ['FLOWHMR_VIDEO_ENCODER'] = str(cfg['video_encoder'])
    for path in (str(APP), str(PACK)):
        if path not in sys.path:
            sys.path.insert(0, path)
    os.chdir(APP)  # official asset / checkpoint paths are repo-relative


def is_oom(exc: BaseException) -> bool:
    text = f'{type(exc).__name__}: {exc}'.lower()
    return 'outofmemory' in text or 'out of memory' in text


# --------------------------------------------------------------------------- #
# official inference
# --------------------------------------------------------------------------- #
def build_engine(cfg: dict):
    from flowhmr.pipeline.v2m_inference import V2MInferenceEngine

    ckpt = APP / 'checkpoints' / 'flowhmr_latest' / 'flowhmr_latest.ckpt'
    model_cfg = APP / 'configs' / 'base' / 'model_basic05B.yml'
    print('=== prepare ===', flush=True)
    print(f'  engine: {ckpt.name} + {model_cfg.name}', flush=True)
    return V2MInferenceEngine.build(
        ckpt=str(ckpt), model_cfg_path=str(model_cfg), device='cuda',
        validation_steps=cfg['steps'], postprocess=True,
    )


def release_engine() -> None:
    import gc

    import torch

    gc.collect()
    torch.cuda.empty_cache()
    try:
        torch.cuda.ipc_collect()
    except Exception:
        pass


def run_inference(engine, video: Path, out_root: Path, cfg: dict) -> tuple[Path, int]:
    """Run the official pipeline; returns (its output dir for this video, frames)."""
    import numpy as np

    stem = video.stem
    state = {'stage': None, 'sam_total': None}

    def progress(stage, percent, desc):
        name = getattr(stage, 'value', str(stage))
        mapped = STAGE_PRINT.get(name)
        if mapped and mapped != state['stage']:
            state['stage'] = mapped
            print(f'\n=== {mapped} ===', flush=True)
        if name == 'feature_extract' and isinstance(percent, int):
            if state['sam_total'] is None:
                state['sam_total'] = 0
                bbox = out_root / stem / f'{stem}_bbox.npz'
                try:
                    with np.load(bbox) as z:
                        state['sam_total'] = int(len(z['bbox']))
                except Exception:
                    pass
            total = state['sam_total']
            if total:
                done = max(0, min(total, int(round(total * percent / 100.0))))
                print(f'[sam extract] {done}/{total} frames', flush=True)

    engine.process_video(
        str(video), str(out_root),
        seed=int(cfg['seed']), cfg_scale=float(cfg['cfg_scale']),
        vggt_frame_interval=int(cfg['vggt_interval']),
        static_camera=bool(cfg['static_camera']),
        sam_chunk_size=int(cfg['sam_chunk_size']),
        sam_hand_stride=int(cfg['sam_hand_stride']),
        sam_hand_seed=int(cfg['sam_hand_seed']),
        progress_callback=progress,
    )

    sub = out_root / stem
    body = sub / f'{stem}_seed{int(cfg["seed"])}.npz'
    if not body.is_file():
        candidates = sorted(sub.glob('*_seed*.npz'))
        if not candidates:
            raise FileNotFoundError(f'official output missing in {sub}')
        body = candidates[-1]
    with np.load(body) as z:
        frames = int(len(z['poses']))
    print(f'GENERATION_DONE {frames}', flush=True)
    return sub, frames


# --------------------------------------------------------------------------- #
# HaMeR + merge
# --------------------------------------------------------------------------- #
def trim_to_track(src: Path, dst: Path, first: int, last: int, total: int) -> None:
    if (last - first + 1) == total:
        shutil.copyfile(src, dst)
        return
    ffmpeg = PACK / 'envs' / 'Lib' / 'site-packages' / 'imageio_ffmpeg' / 'binaries' / 'ffmpeg-win-x86_64-v7.1.exe'
    exe = str(ffmpeg) if ffmpeg.is_file() else 'ffmpeg'
    print(f'  trimming video frames {first}..{last} (of {total})', flush=True)
    subprocess.run(
        [exe, '-y', '-i', str(src),
         '-vf', f"select='between(n\\,{first}\\,{last})',setpts=N/30/TB",
         '-r', '30', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', str(dst)],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def save_correction_payload(job: Path, report: dict, contacts: dict | None) -> None:
    """Persist a contact-correction report (single run or stitched shots)."""
    import numpy as _np

    if not report:
        return
    payload = dict(report)
    payload.setdefault('adapter_version', 1)
    payload.setdefault('source', 'initial')
    payload.setdefault('model_key', 'smplh')
    payload['contact_input'] = ('FlowHMR regressed L ankle/toe, R ankle/toe values clipped to [0,1]; '
                                'not sigmoid logits')
    payload['segment_count'] = len(payload.get('segments') or [])
    payload['shared_floor'] = {'floor': payload.get('floor'), 'coordinate_frame': 'shared_world'}
    if contacts is not None:
        _np.savez(job / 'contacts.npz', **{k: v for k, v in contacts.items() if k != 'handoff_frames'})
        stat = (job / 'contacts.npz').stat()
        payload['contacts_stat'] = [int(stat.st_size), int(stat.st_mtime_ns)]
    save_json(job / 'correction.json', payload)


def plan_shots(engine, video: Path, job: Path, cfg: dict):
    """Run the tracker pass once and plan camera-cut segments."""
    import numpy as np

    sys.path.insert(0, str(TOOLS))
    import multishot

    pre = job / 'official' / 'pre'
    pre.mkdir(parents=True, exist_ok=True)
    working = engine.proc.transcode(str(video), str(pre), force_fps=30)
    tracks_path = job / 'tracks.npz'
    if tracks_path.is_file():
        data = multishot.load_tracks(tracks_path)
        print(f'  shots: cached tracks ({len(data["tracks"])} track ids)', flush=True)
    else:
        result = engine.proc.detect_and_track(str(working))
        tracks = [{'id': t.track_id, 'frames': t.frames, 'bboxes': t.bboxes} for t in result.tracks]
        multishot.save_tracks(tracks_path, tracks, total=result.total_frames,
                              width=result.width, height=result.height)
        data = {'tracks': tracks, 'total_frames': result.total_frames,
                'width': result.width, 'height': result.height}
    segments = multishot.plan_segments(
        data['tracks'], data['total_frames'], data['width'], data['height'],
        min_frames=int(cfg.get('shot_min_frames', 24)))
    plan = multishot.describe_plan(data['tracks'], data['total_frames'],
                                   data['width'], data['height'], segments)
    plan['tracks'] = [dict(row) for row in plan['tracks']]
    save_json(job / 'segments.json', plan)
    print(f'  shots: {len(segments)} segment(s) covering {data["total_frames"]} frames', flush=True)
    for segment in segments:
        print(f"    {segment['start']}-{segment['end']}  {segment['frames']} frames  "
              f"ids {segment['track_ids']}  bbox {segment['bbox_fraction']:.3f}", flush=True)
    return working, plan, data['tracks']


def run_shots(engine, working: Path, plan: dict, tracks: list[dict], job: Path, cfg: dict) -> int:
    """Infer every shot separately, then stitch the SMPL arrays together."""
    import numpy as np

    sys.path.insert(0, str(TOOLS))
    import multishot

    segments = plan['segments']
    total = int(plan['total_frames'])
    width, height = int(plan['width']), int(plan['height'])
    shot_root = job / 'segments'
    shot_out = shot_root / 'official'
    parts, reports, contacts, offsets, clips = [], [], [], [], []

    def progress(stage, percent, desc):
        name = getattr(stage, 'value', str(stage))
        mapped = STAGE_PRINT.get(name)
        if mapped and mapped != progress.stage:
            progress.stage = mapped
            print(f'\n=== {mapped} ===', flush=True)

    progress.stage = None
    for index, segment in enumerate(segments):
        name = f'shot{index:02d}'
        first, last = int(segment['start']), int(segment['end']) - 1
        sub_dir = shot_out / name
        sub_dir.mkdir(parents=True, exist_ok=True)
        clip = shot_root / f'{name}.mp4'
        if not clip.is_file():
            multishot.trim_video(working, clip, first, last)
        # Pre-seed process_video's 30 fps transcode cache so it reuses the clip
        # instead of encoding the same frames again.
        cache = sub_dir / f'{name}_30fps.mp4'
        if not cache.is_file():
            shutil.copyfile(clip, cache)
        clips.append(cache)
        bbox_path = sub_dir / f'{name}_bbox.npz'
        if not bbox_path.is_file():
            boxes = multishot.segment_bbox(tracks, first, last + 1, width, height)
            np.savez(bbox_path, bbox=boxes.astype(np.float32),
                     start_end=np.array([0, boxes.shape[0] - 1], np.int64))
        print(f"\n--- shot {index}: frames {first}..{last} ---", flush=True)
        engine.process_video(
            str(clip), str(shot_out),
            seed=int(cfg['seed']), cfg_scale=float(cfg['cfg_scale']),
            vggt_frame_interval=int(cfg['vggt_interval']),
            static_camera=bool(cfg['static_camera']),
            sam_chunk_size=int(cfg['sam_chunk_size']),
            sam_hand_stride=int(cfg['sam_hand_stride']),
            sam_hand_seed=int(cfg['sam_hand_seed']),
            progress_callback=progress,
        )
        npz_path = sub_dir / f'{name}_seed{int(cfg["seed"])}.npz'
        with np.load(npz_path) as data:
            parts.append({key: data[key] for key in data.files})
        post = getattr(engine, 'postprocess', None)
        report = getattr(post, 'last_report', None)
        plan_contacts = getattr(post, 'last_contacts', None)
        reports.append(json.loads(json.dumps(report, default=float)) if report else {})
        contacts.append(dict(plan_contacts) if plan_contacts else None)
        offsets.append(first)
        print(f'  shot {index}: {len(parts[-1]["poses"])} frames inferred', flush=True)

    merged = multishot.merge_segment_arrays(parts)
    np.savez(job / 'body_postprocess.npz', **merged)
    multishot.concat_videos(clips, job / 'working_30fps.mp4')
    boxes = np.concatenate([multishot.segment_bbox(tracks, segment['start'], segment['end'],
                                                   width, height) for segment in segments], axis=0)
    count = len(boxes)
    np.savez(job / 'main_track.npz', bbox=boxes, bbox_xyxy=boxes,
             detected=np.ones(count, bool),
             start_end=np.array([0, count - 1], np.int64),
             observed_range=np.array([0, count - 1], np.int64))
    report, merged_contacts = multishot.merge_reports(reports, contacts, offsets, count)
    save_correction_payload(job, report, merged_contacts or None)
    print(f'  shots merged: {len(segments)} runs -> {count} frames', flush=True)
    return count


def write_smplh_fbx(job: Path) -> None:
    """Keep the original SMPL-H rig next to the Humanoid52 export."""
    import numpy as _np

    from flowhmr.core.export.smplh2fbx import SMPLH2FBX

    params = {key: value for key, value in _np.load(job / 'final.npz').items()}
    print('  SMPL-H FBX (original rig, kept alongside Humanoid52)', flush=True)
    SMPLH2FBX().convert_npz_to_fbx(params, str(job / 'smplh.fbx'), export_mesh=True)
    shutil.copyfile(job / 'final.npz', job / 'smplh.npz')


def run_hamer(job: Path, cfg: dict) -> None:
    print('\n=== hamer ===', flush=True)
    cmd = [str(PY), '-u', str(TOOLS / 'hamer_pose_sampled.py'),
           '--video', str(job / 'working_30fps.mp4'),
           '--main_track', str(job / 'main_track.npz'),
           '--out_dir', str(job / 'hands'),
           '--detect_interval', str(cfg['hamer_interval']),
           '--batch_size', str(cfg['hamer_batch_size']),
           '--window', str(cfg['hamer_smooth_window']),
           '--max_interp_gap', str(cfg['hamer_max_interp_gap'])]
    log_path = job / 'hamer.log'
    with log_path.open('w', encoding='utf-8', errors='replace') as log:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, encoding='utf-8', errors='replace', bufsize=1)
        try:
            for line in proc.stdout:
                log.write(line)
                log.flush()
                if line.startswith('  [hamer]') or line.startswith('DONE'):
                    print(line.rstrip(), flush=True)
            rc = proc.wait()
        except BaseException:
            proc.kill()
            proc.wait()
            raise
    if rc:
        tail = log_path.read_text(encoding='utf-8', errors='replace')[-4000:]
        raise RuntimeError(f'HaMeR exited {rc}. Tail:\n{tail}')


def validate_fbx(job: Path) -> None:
    import fbx
    import numpy as np
    from scipy.spatial.transform import Rotation

    from flowhmr.core.export.fbx_utils import SMPLH_JOINT2NUM

    with np.load(job / 'final.npz', allow_pickle=False) as z:
        poses = z['poses']
        trans = z['trans']
        n = len(poses)
    with np.load(job / 'body_postprocess.npz', allow_pickle=False) as z:
        assert np.array_equal(poses[:, :22], z['poses'][:, :22]), 'body/wrists changed during merge'
        assert np.array_equal(trans, z['trans']), 'root changed during merge'
    manager = fbx.FbxManager.Create()
    manager.SetIOSettings(fbx.FbxIOSettings.Create(manager, fbx.IOSROOT))
    importer = fbx.FbxImporter.Create(manager, '')
    if not importer.Initialize(str(job / 'final.fbx'), -1, manager.GetIOSettings()):
        raise RuntimeError(importer.GetStatus().GetErrorString())
    scene = fbx.FbxScene.Create(manager, 'validate')
    assert importer.Import(scene)
    importer.Destroy()
    bones, meshes = {}, []

    def walk(node):
        attr = node.GetNodeAttribute()
        if attr:
            typ = attr.GetAttributeType()
            if typ == fbx.FbxNodeAttribute.EType.eSkeleton:
                bones[node.GetName()] = node
            if typ == fbx.FbxNodeAttribute.EType.eMesh:
                meshes.append(node)
        for i in range(node.GetChildCount()):
            walk(node.GetChild(i))

    walk(scene.GetRootNode())
    assert len(bones) == 52 and len(meshes) == 1, (len(bones), len(meshes))
    stack = scene.GetSrcObject(fbx.FbxCriteria.ObjectType(fbx.FbxAnimStack.ClassId), 0)
    layer = scene.GetSrcObject(fbx.FbxCriteria.ObjectType(fbx.FbxAnimLayer.ClassId), 0)
    euler = np.empty((n, 52, 3))
    for name, j in SMPLH_JOINT2NUM.items():
        for axis_i, axis in enumerate('XYZ'):
            curve = bones[name].LclRotation.GetCurve(layer, axis)
            assert curve.KeyGetCount() == n
            assert abs(curve.KeyGetTime(n - 1).GetSecondDouble() - (n - 1) / 30) < 1e-7
            euler[:, j, axis_i] = [curve.KeyGetValue(i) for i in range(n)]
    err = np.degrees((Rotation.from_euler('xyz', euler.reshape(-1, 3), degrees=True).inv() *
                      Rotation.from_rotvec(poses.reshape(-1, 3))).magnitude()).max()
    root = np.stack([[bones['Pelvis'].LclTranslation.GetCurve(layer, axis).KeyGetValue(i)
                      for i in range(n)] for axis in 'XYZ'], axis=-1)
    terr = float(np.max(np.abs((root - root[0]) - (trans - trans[0]) * 100)))
    manager.Destroy()
    save_json(job / 'validation.json', {
        'frames': n, 'fps': 30, 'skeletons': 52, 'meshes': 1,
        'body_root_wrists_preserved': True,
        'rotation_key_max_error_deg': float(err),
        'root_trajectory_key_max_error_cm': terr,
        'fbx_bytes': (job / 'final.fbx').stat().st_size,
        'sdk_reimport': True,
    })


def validate_humanoid52(job: Path) -> dict:
    sys.path.insert(0, str(TOOLS))
    import validate_humanoid52_fbx as validator
    candidates = sorted(job.rglob('*_mixamo.json'))
    if not candidates:
        raise FileNotFoundError('export did not write a *_mixamo.json sidecar')
    meta = json.loads(candidates[0].read_text(encoding='utf-8'))
    frames = int(meta['frames'])
    info = validator.validate(job / 'final.fbx', frames)
    info['body_root_wrists_preserved'] = True
    info['fbx_bytes'] = (job / 'final.fbx').stat().st_size
    return info


def save_correction(job: Path, engine) -> None:
    """Persist the contact-correction report and plan next to the export."""
    import numpy as _np
    post = getattr(engine, 'postprocess', None)
    report = getattr(post, 'last_report', None)
    contacts = getattr(post, 'last_contacts', None)
    if not report:
        return
    payload = dict(report)
    payload.setdefault('adapter_version', 1)
    payload.setdefault('source', 'initial')
    payload.setdefault('model_key', 'smplh')
    payload['options'] = dict(getattr(post, 'options', {}))
    payload['contact_input'] = ('FlowHMR regressed L ankle/toe, R ankle/toe values clipped to [0,1]; '
                                'not sigmoid logits')
    payload['segment_count'] = len(payload.get('segments') or [])
    payload['shared_floor'] = {'floor': payload.get('floor'), 'coordinate_frame': 'shared_world'}
    if contacts is not None:
        _np.savez(job / 'contacts.npz', **{k: v for k, v in contacts.items() if k != 'handoff_frames'})
        stat = (job / 'contacts.npz').stat()
        payload['contacts_stat'] = [int(stat.st_size), int(stat.st_mtime_ns)]
    save_json(job / 'correction.json', payload)


def run_merge(job: Path) -> None:
    print('\n=== merge ===', flush=True)
    subprocess.run([str(PY), '-u', str(TOOLS / 'merge_hamer_smplh.py'),
                    '--body', str(job / 'body_postprocess.npz'),
                    '--hands', str(job / 'hands' / 'working_30fps_hamer_data.pkl'),
                    '--output', str(job / 'final.fbx'),
                    '--npz_only'], check=True)
    # Aligned export path: merged SMPL-H npz -> Humanoid52 rig -> FBX 7400,
    # written inside a toyxyz-compatible project tree (their app has no motion
    # import; retarget only reads its own project layout).
    sys.path.insert(0, str(TOOLS))
    import make_app_project as packer
    request = json.loads((job / 'request.json').read_text(encoding='utf-8'))
    video = Path(request['source'])
    working = job / 'working_30fps.mp4'
    if working.is_file():
        video = working
    settings = load_settings(job / 'settings.json')
    import numpy as _np

    with _np.load(job / 'body_postprocess.npz') as data:
        frames = int(data['num_frames'])
    paths = packer.prepare(job, video, frames=frames, fps=30.0)
    take = paths['take']
    for name in ('correction.json', 'contacts.npz'):
        source = job / name
        if source.is_file():
            shutil.copyfile(source, take / name)
    subprocess.run([str(PY), '-u', str(TOOLS / 'export_humanoid52.py'),
                    str(job / 'final.npz'), '--out', str(take),
                    '--fbx-name', 'final.fbx'], check=True)
    shutil.copyfile(take / 'final.fbx', job / 'final.fbx')
    write_smplh_fbx(job)
    project = packer.finalize(packer.prepare(job, video, frames=frames, fps=30.0))
    print(f"  project: {project['project']}", flush=True)
    validation = validate_humanoid52(job)
    save_json(job / 'validation.json', validation)
    print(f"  FBX validation: {'OK' if not validation['problems'] else validation['problems']}", flush=True)


# --------------------------------------------------------------------------- #
# one video
# --------------------------------------------------------------------------- #
def run_video(video_path: str, root: Path, cfg: dict, fresh: bool) -> Path:
    video = Path(video_path).expanduser().resolve()
    if not video.is_file():
        raise FileNotFoundError(f'video not found: {video}')
    job = root / job_name(video, cfg)
    if fresh and job.exists():
        shutil.rmtree(job)
    job.mkdir(parents=True, exist_ok=True)
    save_json(job / 'settings.json', cfg)
    save_json(job / 'request.json', {'source': str(video), 'created': time.time()})
    print(f'Output: {job}', flush=True)
    print(f'  source: {video}', flush=True)
    print(f'  hand stride {cfg["sam_hand_stride"]} | sam chunk {cfg["sam_chunk_size"]} | '
          f'static camera {bool(cfg["static_camera"])} | hamer N={cfg["hamer_interval"]}', flush=True)

    started = time.perf_counter()
    official_root = job / 'official'
    engine = build_engine(cfg)
    combined = 0
    try:
        plan = shots = None
        if cfg.get('multi_shot', True):
            working_full, plan, shots = plan_shots(engine, video, job, cfg)
        if plan and len(plan['segments']) > 1:
            combined = run_shots(engine, working_full, plan, shots, job, cfg)
        else:
            sub, frames = run_inference(engine, video, official_root, cfg)
            save_correction(job, engine)
    finally:
        engine = None
        release_engine()

    peak_inference = 0.0
    try:
        import torch
        peak_inference = torch.cuda.max_memory_allocated() / 2 ** 30
    except Exception:
        pass
    save_json(job / 'gpu_peak.json', {'inference_GiB': round(peak_inference, 2),
                                      'peak_alloc_GiB': round(peak_inference, 2),
                                      'updated': time.time()})

    if combined:
        frames = combined
    else:
        stem = video.stem
        body_npz = sub / f'{stem}_seed{int(cfg["seed"])}.npz'
        working = sub / f'{stem}_30fps.mp4'
        bbox_npz = sub / f'{stem}_bbox.npz'
        for p in (body_npz, working, bbox_npz):
            if not p.is_file():
                raise FileNotFoundError(f'official output missing: {p}')

        import cv2
        import numpy as np

        with np.load(bbox_npz) as z:
            boxes = z['bbox'].astype(np.float32)
            start_end = z['start_end'] if 'start_end' in z.files else np.array([0, len(boxes) - 1])
        cap = cv2.VideoCapture(str(working))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        trim_to_track(working, job / 'working_30fps.mp4', int(start_end[0]), int(start_end[1]), total)
        n = len(boxes)
        np.savez(job / 'main_track.npz', bbox=boxes, bbox_xyxy=boxes, detected=np.ones(n, bool),
                 start_end=start_end, observed_range=start_end)
        shutil.copyfile(body_npz, job / 'body_postprocess.npz')

    run_hamer(job, cfg)
    peak_hamer = 0.0
    try:
        info = json.loads((job / 'hands' / 'run_info.json').read_text(encoding='utf-8'))
        peak_hamer = float(info.get('peak_alloc_GiB') or 0.0)
    except Exception:
        pass
    peak_total = max(peak_inference, peak_hamer)
    save_json(job / 'gpu_peak.json', {'inference_GiB': round(peak_inference, 2),
                                      'hamer_GiB': round(peak_hamer, 2),
                                      'peak_alloc_GiB': round(peak_total, 2),
                                      'updated': time.time()})
    run_merge(job)

    elapsed = time.perf_counter() - started
    save_json(job / 'run_manifest.json', {
        'status': 'completed', 'elapsed_seconds': elapsed, 'frames': frames,
        'peak_alloc_GiB': round(max(peak_inference, peak_hamer), 2),
        'peak_inference_GiB': round(peak_inference, 2), 'peak_hamer_GiB': round(peak_hamer, 2),
        'final_fbx': str(job / 'final.fbx'), 'final_npz': str(job / 'final.npz'),
        'settings': cfg, 'source': str(video),
    })
    print(f'\nDONE: {job / "final.fbx"}\nData: {job / "final.npz"}\nTime: {elapsed:.1f}s', flush=True)
    return job


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('videos', nargs='*')
    ap.add_argument('--settings', default=str(PACK / 'pipeline_settings.json'))
    ap.add_argument('--output-root', default=str(PACK / 'outputs' / 'video_to_fbx'))
    ap.add_argument('--fresh', action='store_true')
    args = ap.parse_args()
    if not args.videos:
        ap.error('Drop a video onto the BAT, or pass one or more video paths')

    cfg = load_settings(Path(args.settings))
    configure(cfg)
    root = Path(args.output_root).resolve()
    root.mkdir(parents=True, exist_ok=True)

    failed = []
    for video in args.videos:
        try:
            run_video(video, root, cfg, args.fresh)
        except Exception as exc:
            traceback.print_exc()
            failed.append(str(video))
            if is_oom(exc):
                print(f'[OOM] {Path(video).name}: {type(exc).__name__}: {str(exc).splitlines()[0]}', flush=True)
                print('     显存或系统内存不足 / out of VRAM or system commit memory', flush=True)
            print(f'FAILED: {video}: {exc}', flush=True)
    if failed:
        print(f'Failed {len(failed)}/{len(args.videos)} input(s)', flush=True)
        return 1
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
    except Exception:
        traceback.print_exc()
        sys.exit(1)
