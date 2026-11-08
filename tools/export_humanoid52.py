"""Export a merged SMPL-H job to the toyxyz_RGBMocap Humanoid52 rig.

This is the aligned export path: SMPL-H parameters -> Humanoid52 rig (T-pose
rest, per-bone constant basis, Mixamo bone names, 13 terminal markers) ->
binary FBX 7400 with linear keys, plus the rig/BVH assets the original
application consumes.

The algorithm modules live in app/flowhmr/third_party/Humanoid52 and are a
verbatim vendor of toyxyz_RGBMocap (see logs/UPSTREAM.md).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

PACK = Path(__file__).resolve().parents[1]
APP = PACK / 'app' / 'flowhmr'
sys.path.insert(0, str(APP))
sys.path.insert(0, str(APP / 'third_party'))
sys.path.insert(0, str(PACK))

import torch  # noqa: E402

from flowhmr.core.bodymodels.smpl_skeleton import SMPLMesh, SMPLSkeleton  # noqa: E402
from flowhmr.core.math.geometry import (  # noqa: E402
    angle_axis_to_rotation_matrix,
    rot6d_to_rotation_matrix,
    rotation_matrix_to_rot6d,
)

from Humanoid52 import bvh_bind, fbx_bind, fbx_writer, source_bvh, source_surface  # noqa: E402
from Humanoid52.flow_export import proper_rotations, target_rig  # noqa: E402
from Humanoid52.mixamo_rig import fk, mannequin  # noqa: E402
from Humanoid52.source_animation import fit, policy, temporal_halo, validate  # noqa: E402

MODEL = APP / 'assets' / 'body_models' / 'smplh' / 'neutral' / 'model.npz'


def load_params(path: Path, fps: float = 30.0):
    """Return (rot6d (T,52,6), trans (T,3), shapes (1,16), fps).

    Accepts the merged job npz (axis-angle poses) or a toyxyz/RGBMocap
    parameter ``.pt`` (already rot6d) so a foreign project can be re-exported.
    """
    if path.suffix.lower() == '.pt':
        payload = torch.load(path, map_location='cpu', weights_only=False)
        rot6d = torch.as_tensor(payload['rot6d'], dtype=torch.float32)[0]
        trans = torch.as_tensor(payload['trans'], dtype=torch.float32)[0]
        shapes = torch.as_tensor(payload['shapes'], dtype=torch.float32).reshape(1, 16)
        return rot6d.contiguous(), trans.contiguous(), shapes, fps
    with np.load(path) as data:
        poses = torch.from_numpy(np.asarray(data['poses'], dtype=np.float32))
        trans = torch.from_numpy(np.asarray(data['trans'], dtype=np.float32))
        betas = torch.from_numpy(np.asarray(data['betas'], dtype=np.float32)).reshape(1, 16)
        if 'mocap_framerate' in data.files:
            fps = float(data['mocap_framerate'])
    if poses.ndim != 3 or poses.shape[1:] != (52, 3):
        raise ValueError(f'Expected poses (T,52,3), got {tuple(poses.shape)}')
    rot_mats = angle_axis_to_rotation_matrix(poses.reshape(-1, 3)).reshape(len(poses), 52, 3, 3)
    return rotation_matrix_to_rot6d(rot_mats).contiguous(), trans, betas, fps


def build_rig():
    mesh = SMPLMesh(model_path=str(MODEL)).eval()
    body = SMPLSkeleton(model_path=str(MODEL)).eval()
    return mesh, body


def neutral_anatomy(body, mesh, betas):
    neutral = dict(rot_mats=torch.eye(3).reshape(1, 1, 3, 3).expand(1, 52, 3, 3),
                   shapes=betas, trans=torch.zeros(1, 3))
    with torch.no_grad():
        rest = body(neutral)['keypoints3d'][0].numpy()
        template = mesh(neutral)['vertices'][0].numpy()
    return rest, template


def compute_surface(mesh, rot6d, betas, trans, path: Path, emit):
    frames = int(rot6d.shape[0])
    surface = np.lib.format.open_memmap(path, mode='w+', dtype=np.float32, shape=(frames, 6890, 3))
    started = time.perf_counter()
    with torch.no_grad():
        for at in range(0, frames, 8):
            stop = min(frames, at + 8)
            out = mesh(dict(rot6d=rot6d[at:stop], shapes=betas, trans=trans[at:stop]))['vertices']
            surface[at:stop] = out.numpy()
    surface.flush()
    emit(f'  surface: {frames} frames in {time.perf_counter() - started:.1f}s -> {path.name}')
    return surface


def fit_rig(rig, mapping, binding, body, params, endpoints, fps, chunk, emit):
    rot6d, betas, trans = params
    frames = int(rot6d.shape[0])
    parents = body.parents.tolist()
    local_all = np.empty((frames, 52, 3, 3), np.float32)
    world_all = np.empty((frames, 52, 3, 3), np.float32)
    posed_all = np.empty((frames, 52, 3), np.float32)
    roots = np.empty((frames, 3), np.float32)
    native_points = np.empty((frames, 22, 3), np.float32)
    native_rotations = np.empty((frames, 22, 3, 3), np.float32)
    errors = []
    finger_errors = []
    distal_error = 0.0
    hinge_error = 0.0
    halo = temporal_halo(fps)
    started = time.perf_counter()
    with torch.inference_mode():
        for first in range(0, frames, chunk):
            end = min(frames, first + chunk)
            begin = max(0, first - halo)
            stop = min(frames, end + halo)
            selected = dict(rot6d=rot6d[begin:stop], shapes=betas, trans=trans[begin:stop])
            joints = body(selected)['keypoints3d'].numpy()
            local_native = proper_rotations(rot6d_to_rotation_matrix(selected['rot6d']).numpy())
            world_native = local_native.copy()
            for j, p in enumerate(parents):
                if p >= 0:
                    world_native[:, j] = world_native[:, p] @ world_native[:, j]
            local, posed, world = fit(
                rig, joints[:, 0], world_native[:, :22], joints[:, :22],
                surface_endpoints=endpoints[begin:stop], fps=fps)
            for j in range(22, 52):
                p = rig['parents'][j]
                world[:, j] = world_native[:, mapping[j]]
                local[:, j] = world[:, p].swapaxes(-1, -2) @ world[:, j]
            posed, world = fk(rig, joints[:, 0], local)
            core = slice(first - begin, end - begin)
            local_core = local[core]
            world_core = world[core]
            posed_core = posed[core]
            joints_core = joints[core]
            local_all[first:end] = local_core
            world_all[first:end] = world_core
            posed_all[first:end] = posed_core
            roots[first:end] = joints_core[:, 0]
            native_points[first:end] = joints_core[:, :22]
            native_rotations[first:end] = world_native[core, :22]
            errors.append(np.linalg.norm(posed_core[:, :22] - joints_core[:, :22], axis=-1))
            finger_errors.append(np.linalg.norm(posed_core[:, 22:] - joints_core[:, mapping[22:]], axis=-1))
            for j in (7, 8, 20, 21):
                distal_error = max(distal_error, float(np.abs(world_core[:, j] - world_native[core, j]).max()))
            hinge_error = max(hinge_error, float(validate(local_core)))
            if (first // chunk) % 4 == 0:
                emit(f'  fit: {end}/{frames} frames  {time.perf_counter() - started:.0f}s')
    error = np.concatenate(errors)
    finger_error = np.concatenate(finger_errors)
    stats = dict(mean_m=float(error.mean()), p95_m=float(np.quantile(error, .95)), max_m=float(error.max()),
                 finger_fit_mean_m=float(finger_error.mean()),
                 finger_fit_max_m=float(finger_error.max()),
                 distal_rotation_max_error=float(distal_error),
                 hinge_axis_max_error=float(hinge_error))
    return local_all, world_all, posed_all, roots, native_points, native_rotations, stats


def load_foot_contacts(path: Path, frames: int):
    if path.suffix.lower() == '.pt':
        payload = torch.load(path, map_location='cpu', weights_only=False)
        contacts = payload.get('foot_contacts')
        return None if contacts is None else torch.as_tensor(contacts, dtype=torch.float32)
    with np.load(path) as data:
        if 'foot_contacts' not in data.files:
            return None
        return torch.from_numpy(np.asarray(data['foot_contacts'], dtype=np.float32))


def export(npz_path: Path, out_dir: Path, *, chunk: int = 64, write_assets: bool = True,
           fps: float = 30.0, fbx_name: str = 'motion.fbx', write_surface: bool = True,
           emit=print) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    rot6d, trans, betas, fps = load_params(npz_path, fps)
    foot_contacts = load_foot_contacts(npz_path, int(rot6d.shape[0]))
    import uuid
    kind = 'corrected_initial' if (out_dir / 'correction.json').is_file() else 'initial'
    token = uuid.uuid4().hex[:12]
    stem = f'{kind}_flow-{token}'
    frames = int(rot6d.shape[0])
    emit(f'  input: {npz_path.name}  {frames} frames @ {fps:g} fps')

    mesh, body = build_rig()
    rest, template = neutral_anatomy(body, mesh, betas)
    rig, mapping = target_rig(rest, template, mesh.lbs_weights.numpy())
    binding = bvh_bind.bind_pose(rig)
    fbx_binding = fbx_bind.prepare(rig, rest, template, 'smplh', fps)
    emit(f'  rig: {len(binding["names"])} bones, rest/tip basis ready')

    surface_path = out_dir / (f'{stem}-surface.npy' if write_surface else '_surface_tmp.npy')
    surface = compute_surface(mesh, rot6d, betas, trans, surface_path, emit)
    endpoints = source_surface.native_endpoints(surface)

    local, world, posed, roots, native_points, native_rotations, fit_stats = fit_rig(
        rig, mapping, binding, body, (rot6d, betas, trans), endpoints, fps, chunk, emit)
    emit(f'  fit error: mean {fit_stats["mean_m"] * 1000:.1f} mm, '
         f'p95 {fit_stats["p95_m"] * 1000:.1f} mm, max {fit_stats["max_m"] * 1000:.1f} mm')

    frames44 = np.tile(np.eye(4, dtype=np.float64), (frames, 52, 1, 1))
    frames44[:, :, :3, :3] = world @ binding['basis']
    frames44[:, :, :3, 3] = posed

    fbx_path = out_dir / fbx_name
    if fbx_path.exists():
        fbx_path.unlink()
    t0 = time.perf_counter()
    fbx_writer.write(
        str(fbx_path), binding['names'], binding['parents'],
        fbx_binding['fbx_rest'], fbx_binding['fbx_rest_frames'],
        fbx_binding['fbx_display_frames'], fbx_binding['fbx_tip_local'],
        frames44, fps)
    emit(f'  fbx: {fbx_path.name}  {fbx_path.stat().st_size / 2 ** 20:.2f} MB  '
         f'{time.perf_counter() - t0:.1f}s')

    if write_assets:
        import hashlib
        from Humanoid52.bvh_bind import VERSION as BIND_VERSION
        from Humanoid52.mixamo_rig import VERSION as RIG_VERSION
        from Humanoid52.source_animation import FIT_VERSION

        def _stat(path: Path):
            value = Path(path).stat()
            return [int(value.st_size), int(value.st_mtime_ns)]

        def _sha256(path: Path):
            digest = hashlib.sha256()
            with open(path, 'rb') as handle:
                for chunk in iter(lambda: handle.read(1 << 20), b''):
                    digest.update(chunk)
            return digest.hexdigest()

        rig_path = out_dir / f'{stem}-rig.npy'
        reference_path = out_dir / f'{stem}-reference.npy'
        rig_joints_path = out_dir / f'{stem}-rig-joints.npy'
        asset_path = out_dir / f'{stem}-rig-asset.npz'
        bvh_path = out_dir / f'{stem}.bvh'
        surface_out = out_dir / f'{stem}-surface.npy'
        surface_asset_out = out_dir / f'{stem}-surface-asset.npz'
        if write_surface:
            surface.flush()
            from Humanoid52.mesh_data import adjacency
            faces = np.asarray(mesh.faces, np.uint32)
            offsets, neighbors = adjacency(faces, 6890)
            np.savez(surface_asset_out, faces=faces, offsets=offsets, neighbors=neighbors,
                     vertex_count=np.array(6890))
        np.save(rig_path, frames44.astype(np.float32))
        np.save(reference_path, native_points.astype(np.float32))
        np.save(rig_joints_path, posed[:, :22].astype(np.float32))
        dummy = mannequin(rig)
        np.savez_compressed(
            asset_path,
            packed=bvh_bind.asset_frames(dummy['packed'], binding), faces=dummy['faces'],
            vertex_count=np.array(dummy['vertex_count']), rest=binding['rest'],
            parents=binding['parents'], ends=binding['ends'], export_basis=binding['basis'],
            anatomical_calibration=rig['alignment'], **fbx_binding)
        bvh_info = source_bvh.write_bvh(str(bvh_path), rig, roots, local, fps)
        shutil.copyfile(bvh_path, out_dir / f'{kind}_motion.bvh')
        shutil.copyfile(bvh_path, out_dir / 'motion.bvh')
        joints_path = out_dir / f'{kind}_joints.npz'
        np.savez_compressed(joints_path, joints=native_points,
                            rotations=native_rotations, fps=np.array(fps))
        params_path = out_dir / f'{kind}.pt'
        torch.save(dict(
            rot6d=rot6d.unsqueeze(0).contiguous(), shapes=betas.reshape(1, 1, 16).contiguous(),
            trans=trans.unsqueeze(0).contiguous(),
            root_rot6d=rot6d[:, 0:1].contiguous(), body_rot6d=rot6d[:, 1:].contiguous(),
            foot_contacts=(foot_contacts if foot_contacts is not None
                           else torch.zeros(1, len(rot6d), 4)),
            scene_offset=torch.zeros(3), source_representation='smplh',
            flow_version=1, placement_version=2), params_path)
        correction_payload = None
        correction_path = out_dir / 'correction.json'
        if correction_path.is_file():
            try:
                correction_payload = json.loads(correction_path.read_text(encoding='utf-8'))
            except Exception as exc:
                emit(f'  correction.json unreadable ({type(exc).__name__}); exporting null')
        contacts_src = out_dir / 'contacts.npz'
        if contacts_src.is_file():
            shutil.copyfile(contacts_src, out_dir / f'{kind}_contacts.npz')
            shutil.copyfile(contacts_src, out_dir / 'contacts.npz') if False else None
        code_dir = APP / 'third_party' / 'Humanoid52'
        code_sources = sorted(p for p in code_dir.glob('*.py') if not p.name.startswith('_'))
        code = {p.name: _sha256(p) for p in code_sources}
        code_stats = {p.name: _stat(p) for p in code_sources}
        outputs = {}
        for path in (rig_path, reference_path, rig_joints_path, asset_path, bvh_path,
                     joints_path, params_path):
            outputs[path.name] = _stat(path)
        meta = dict(
            version=RIG_VERSION, fit_version=FIT_VERSION, kind=kind,
            representation='smplh', model_key='smplh', model_path=str(MODEL),
            geometry='Native SMPL-H / anatomical Humanoid52',
            method='Paired joint front/side surface roll with proximal confidence fallback',
            source_model='flowhmr', frames=frames, fps=fps,
            names=list(binding['names']), parents=binding['parents'].tolist(),
            frames_file=rig_path.name, joints_file=joints_path.name,
            reference_joints_file=reference_path.name,
            asset_file=asset_path.name, bvh_file=bvh_path.name,
            raw_file=(surface_out.name if write_surface else None),
            raw_asset=(surface_asset_out.name if write_surface else None),
            raw_vertex_count=6890,
            fbx_file=fbx_path.name, params_file=params_path.name,
            source_fit=fit_stats, selected_source_fit=fit_stats,
            finger_fit_mean_m=fit_stats['finger_fit_mean_m'],
            finger_fit_max_m=fit_stats['finger_fit_max_m'],
            distal_rotation_max_error=fit_stats['distal_rotation_max_error'],
            hinge_axis_max_error=fit_stats['hinge_axis_max_error'],
            hinge_reference='Surface-aligned bind and animation; multi-axis residuals retained to preserve source pose',
            terminal_anatomy='Registered SMPL-H fingertips/toes and shaped head envelope',
            anatomical_limits_deg=None, correction=correction_payload,
            code=code, code_stats=code_stats, model_stat=_stat(MODEL),
            temporal_fit=dict(policy(fps)),
            export_bind=dict(
                version=BIND_VERSION, length_unit='meter', up_axis='Y',
                root_offset=binding['rest'][0].tolist(), root_bone='Hips',
                marker_bones=int(sum(1 for j in range(52)
                                      if j not in set(binding['parents'].tolist()))),
                fbx_version=7400, interpolation='LINEAR', unit_scale_factor=100.0,
                rotation_order_default='Per joint',
                joint_rotation_orders={name: ''.join(order)
                                       for name, order in zip(binding['names'], bvh_info['orders'])},
                minimum_singularity_margin=bvh_info.get('minimum_singularity_margin')),
            outputs=outputs,
            sources={str(npz_path): _stat(npz_path), str(MODEL): _stat(MODEL),
                     joints_path.name: _stat(joints_path)},
            source=str(npz_path))
        if correction_payload is not None:
            payload = dict(correction_payload)
            payload.pop('segments', None)
            payload.setdefault('elapsed', payload.pop('elapsed_seconds', None))
            payload.setdefault('contact_revision', None)
            payload.setdefault('model_stat', _stat(MODEL))
            payload.setdefault('output_stat', _stat(npz_path))
            for key in ('contact_sole_height_before_m', 'contact_sole_height_after_m'):
                payload.setdefault(key, None)
            native_dir = PACK / 'Lib' / 'site-packages' / 'motion_correction'
            native = sorted(native_dir.glob('*.pyd'))
            payload['backend_sha256'] = _sha256(native[0]) if native else None
            rgb_dir = PACK / 'Lib' / 'site-packages' / 'rgbmocap_correction'
            rgb_sources = sorted(p for p in rgb_dir.glob('*.py'))
            payload['code'] = {p.name: _sha256(p) for p in rgb_sources}
            payload['contacts_stat'] = _stat(contacts_src) if contacts_src.is_file() else None
            meta['correction'] = payload
        (out_dir / f'{kind}_mixamo.json').write_text(
            json.dumps(meta, indent=2, ensure_ascii=False), encoding='utf-8')
        (out_dir / f'{kind}_sources.json').write_text(
            json.dumps({str(npz_path): _stat(npz_path)}, indent=2), encoding='utf-8')
        (out_dir / f'{kind}_artifacts.json').write_text(json.dumps(dict(
            version=1, source_stat=_stat(npz_path), fps=fps, model_stat=_stat(MODEL),
            outputs=outputs), indent=2), encoding='utf-8')
        if write_surface:
            (out_dir / f'{kind}_mesh.json').write_text(json.dumps(dict(
                version=1, representation='smplh', frames_file=surface_out.name,
                asset=surface_asset_out.name,
                source_size=_stat(npz_path)[0], source_mtime_ns=_stat(npz_path)[1],
                model_size=_stat(MODEL)[0], model_mtime_ns=_stat(MODEL)[1]),
                indent=2), encoding='utf-8')
            outputs[surface_out.name] = _stat(surface_out)
            outputs[surface_asset_out.name] = _stat(surface_asset_out)
        if correction_payload is not None:
            (out_dir / str(kind + '_correction.json')).write_text(
                json.dumps(correction_payload, indent=2, ensure_ascii=False), encoding='utf-8')
            (out_dir / 'correction_report.json').write_text(json.dumps(
                {'version': 1, 'results': {kind: correction_payload}},
                indent=2, ensure_ascii=False), encoding='utf-8')
        (out_dir / 'export.json').write_text(json.dumps(dict(
            format='FBX', frames=frames, fps=fps, result_kind=kind,
            skeleton='Humanoid52', length_unit='meter', up_axis='Y',
            rotation_order='Per joint', joints=list(binding['names']),
            source='FlowHMR native SMPL-H',
            finger_motion='Native FlowHMR animated hand rotations',
            bind_pose=dict(root_offset=binding['rest'][0].tolist()),
            fbx=dict(file=fbx_path.name, version=7400, interpolation='LINEAR',
                     unit_scale_factor=100.0)), indent=2, ensure_ascii=False), encoding='utf-8')
        emit(f'  assets: {kind}*-rig.npy / -reference.npy / -rig-joints.npy / -rig-asset.npz / .bvh / '
             f'{kind}_joints.npz / {kind}.pt / {kind}_mixamo.json')
    surface._mmap.close()
    if not write_surface:
        surface_path.unlink(missing_ok=True)
    return dict(fbx=str(fbx_path), frames=frames, fps=fps, fit=fit_stats)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('npz', help='Merged SMPL-H npz (poses/trans/betas)')
    ap.add_argument('--out', required=True, help='Output directory')
    ap.add_argument('--chunk', type=int, default=64)
    ap.add_argument('--fps', type=float, default=30.0)
    ap.add_argument('--fbx-name', default='motion.fbx')
    ap.add_argument('--no-surface', action='store_true',
                    help='Skip the raw SMPL surface (the app retarget path reads it)')
    ap.add_argument('--no-assets', action='store_true', help='Only write the FBX')
    args = ap.parse_args()
    started = time.perf_counter()
    info = export(Path(args.npz), Path(args.out), chunk=args.chunk,
                  write_assets=not args.no_assets, fps=args.fps, fbx_name=args.fbx_name,
                  write_surface=not args.no_surface)
    info['seconds'] = round(time.perf_counter() - started, 1)
    print(json.dumps(info, indent=2, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
