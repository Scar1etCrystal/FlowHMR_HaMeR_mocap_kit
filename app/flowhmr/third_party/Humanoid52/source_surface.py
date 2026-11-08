"""Bounded selected-source surface sampling for export, never during painting."""
from pathlib import Path
import json
import numpy as np
from .surface_landmarks import PATCHES


def layout(topology):
    patches = PATCHES[topology]
    groups = ('front', 'back', 'side_a', 'side_b')
    ids = np.array(sorted({i for p in patches for g in groups for i in p[g]}))
    lookup = [[np.searchsorted(ids, p[g]) for g in groups] for p in patches]
    return ids, lookup


def endpoints(vertices, lookup):
    result = np.empty((len(vertices), 8, 4, 3), dtype=float)
    for k, patch in enumerate(lookup):
        for j, indices in enumerate(patch):
            result[:, k, j] = vertices[:, indices].mean(axis=1)
    if not np.isfinite(result).all():
        raise ValueError('Non-finite selected source surface')
    return result


def native_endpoints(surface):
    """SMPL-H surface is already generated in bounded batches by its worker."""
    if surface.ndim != 3 or surface.shape[1:] != (6890, 3):
        raise ValueError('Expected native SMPL-H surface')
    ids, lookup = layout('smplh')
    result = np.empty((len(surface), 8, 4, 3), dtype=float)
    for start in range(0, len(surface), 128):
        result[start:start+128] = endpoints(np.asarray(surface[start:start+128, ids], float), lookup)
    return result


def smplx_endpoints(project, kind, count):
    """Exact selected preview LBS at the fixed patches, with receipt inputs."""
    from .result_artifacts import file_stat
    project = Path(project)
    metadata = project / (kind + '_mesh.json')
    info = json.loads(metadata.read_text(encoding='utf-8'))
    if Path(info['frames_file']).name != info['frames_file']:
        raise ValueError('Invalid source surface frame path')
    from .project_storage import resolve_reference
    asset = resolve_reference(project, info['asset'])
    frames_path = project / info['frames_file']
    sources = {str(p): file_stat(p) for p in (metadata, asset, frames_path)}
    ids, lookup = layout('smplx')
    with np.load(asset, allow_pickle=False) as data:
        n = int(data['vertex_count'])
        if n != 10475:
            raise ValueError('Expected registered SMPL-X topology')
        packed = data['packed']
        template = packed[:n*3].reshape(n, 3)[ids].astype(float)
        shapes = packed[n*3:n*63].reshape(20, n, 3)[:, ids].astype(float)
        poses = packed[n*63:n*1521].reshape(486, n, 3)[:, ids].astype(float)
        weights = packed[n*1521:].reshape(n, 55)[ids].astype(float)
    frames = np.load(frames_path, mmap_mode='r', allow_pickle=False)
    try:
        if frames.shape != (count, 1389):
            raise ValueError('Selected source surface frame count mismatch')
        result = np.empty((count, 8, 4, 3), dtype=float)
        for start in range(0, count, 128):
            f = np.asarray(frames[start:start+128], float)
            posed = template + np.einsum('tb,bvc->tvc', f[:,1366:1386], shapes) + np.einsum('tp,pvc->tvc', f[:,880:1366], poses)
            transform = np.einsum('vj,tjk->tvk', weights, f[:,:880].reshape(-1,55,16)).reshape(-1,len(ids),4,4)
            vertices = np.einsum('tvij,tvj->tvi', transform[:,:,:3,:3], posed) + transform[:,:,:3,3] + f[:,-3:,None].transpose(0,2,1)
            result[start:start+len(f)] = endpoints(vertices, lookup)
    finally:
        frames._mmap.close()
    if any(file_stat(Path(p)) != stamp for p, stamp in sources.items()):
        raise ValueError('Selected source surface changed during sampling')
    return result, sources
