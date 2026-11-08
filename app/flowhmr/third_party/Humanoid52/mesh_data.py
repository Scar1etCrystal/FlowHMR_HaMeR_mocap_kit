"""Prepare exact SMPL-X LBS inputs outside the GUI; never bake a whole mesh video.

The local model-derived asset retains the registered model's usage conditions.
It is created only from the user's imported body model, never bundled publicly.
"""
from pathlib import Path
import os
import json
import numpy as np
from .storage import atomic_json

MESH_VERSION = 1
FRAME_FLOATS = 55 * 16 + 54 * 9 + 20 + 3


def adjacency(faces, count):
    """CSR list of oriented triangle neighbors, for area-weighted smooth normals."""
    centers = faces.reshape(-1)
    pairs = np.stack((np.roll(faces, -1, axis=1), np.roll(faces, -2, axis=1)), axis=-1).reshape(-1, 2)
    order = np.argsort(centers, kind="stable")
    offsets = np.r_[0, np.cumsum(np.bincount(centers, minlength=count))].astype(np.uint32)
    return offsets, pairs[order].astype(np.uint32).reshape(-1)


def prepare_mesh_preview(project, model_path=None, kinds=None):
    import torch
    import smplx
    from smplx.lbs import batch_rodrigues, batch_rigid_transform
    from .paths import MODELS
    project = Path(project)
    from .mamma_body import body_path
    model_path = Path(model_path) if model_path is not None else body_path(project)
    body = smplx.SMPLX(str(model_path), gender="neutral", num_pca_comps=12,
                      flat_hand_mean=False, batch_size=1).eval()
    stat = model_path.stat()
    # One static model asset, independent of the recording length.
    asset = model_path.parent / f".mesh-lbs-v{MESH_VERSION}-{stat.st_size}-{stat.st_mtime_ns}.npz"
    if not asset.exists():
        shape_dirs = torch.cat((body.shapedirs, body.expr_dirs), dim=-1)
        packed = np.concatenate((body.v_template.numpy().reshape(-1),
            shape_dirs.numpy().transpose(2,0,1).reshape(-1),
            body.posedirs.numpy().reshape(-1), body.lbs_weights.numpy().reshape(-1))).astype(np.float32)
        offsets, neighbors = adjacency(body.faces, len(body.v_template))
        temporary = asset.with_suffix(".partial.npz")
        np.savez(temporary, packed=packed, faces=body.faces.astype(np.uint32),
                 offsets=offsets, neighbors=neighbors, vertex_count=np.array(len(body.v_template)))
        os.replace(temporary, asset)
    from .motion_tools import RESULTS
    for kind, (filename, key) in RESULTS.items():
        if kinds is not None and kind not in kinds:
            continue
        source = project / filename
        if not source.exists():
            continue
        source_stat = source.stat()
        target = project / f"{kind}_mesh-v{MESH_VERSION}-{source_stat.st_size}-{source_stat.st_mtime_ns}-{stat.st_mtime_ns}.npy"
        metadata = project / f"{kind}_mesh.json"
        if metadata.exists():
            info = json.loads(metadata.read_text(encoding="utf-8"))
            cached=Path(info.get('frames_file',''))
            if (cached.name==str(cached) and (project/cached).is_file() and
                info.get('version')==MESH_VERSION and
                (info.get('source_size'),info.get('source_mtime_ns'))==(source_stat.st_size,source_stat.st_mtime_ns) and
                (info.get('model_size'),info.get('model_mtime_ns'))==(stat.st_size,stat.st_mtime_ns)):
                continue
        if target.exists():
            # An invalid/incomplete publication may still be pinned by an old
            # viewer. Publish a new immutable path instead of overwriting it.
            import uuid
            target=target.with_name(target.stem+'-'+uuid.uuid4().hex[:12]+'.npy')
        params = torch.load(source, map_location="cpu", weights_only=False)[key]
        length = len(params["body_pose"])
        temporary = target.with_suffix(".partial.npy")
        output = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.float32, shape=(length, FRAME_FLOATS))
        with torch.no_grad():
            for first in range(0, length, 256):
                count = min(256, length-first)
                full = body.pose_mean.reshape(1,55,3).expand(count,-1,-1).clone()
                full[:,0] += params["global_orient"][first:first+count]
                full[:,1:22] += params["body_pose"][first:first+count].reshape(count,21,3)
                for field,index in (("jaw_pose",22),("leye_pose",23),("reye_pose",24)):
                    if field in params:
                        full[:,index] += params[field][first:first+count]
                for field,begin,components in (("left_hand_pose",25,body.left_hand_components),
                                              ("right_hand_pose",40,body.right_hand_components)):
                    if field in params:
                        hand = params[field][first:first+count]
                        if hand.shape[-1] == 12:
                            hand = hand @ components
                        full[:,begin:begin+15] += hand.reshape(count,15,3)
                beta = params["betas"][first:first+count]
                expression = params.get("expression")
                expression = expression[first:first+count] if expression is not None else torch.zeros(count,10)
                shapes = torch.cat((beta,expression),dim=-1)
                dirs = torch.cat((body.shapedirs,body.expr_dirs),dim=-1)
                shaped = body.v_template + torch.einsum("bi,vci->bvc",shapes,dirs)
                rest = torch.einsum("jv,bvc->bjc",body.J_regressor,shaped)
                rotations = batch_rodrigues(full.reshape(-1,3)).reshape(count,55,3,3)
                _, transforms = batch_rigid_transform(rotations,rest,body.parents)
                features = (rotations[:,1:] - torch.eye(3)).reshape(count,-1)
                output[first:first+count] = torch.cat((transforms.reshape(count,-1),features,shapes,
                                                       params["transl"][first:first+count]),dim=-1).numpy()
        output.flush()
        output._mmap.close()
        os.replace(temporary,target)
        state = source.stat()
        atomic_json(project / f"{kind}_mesh.json", {"version":MESH_VERSION,"asset":str(asset.resolve()),
            "frames_file":target.name,
            "frames":length,"source_size":state.st_size,"source_mtime_ns":state.st_mtime_ns,
            "model_size":stat.st_size,"model_mtime_ns":stat.st_mtime_ns,
            "geometry":"Full SMPL-X LBS with shape and pose correctives; no retargeting or geometry smoothing",
            "shading":"Area-weighted shared vertex normals"})
        # Only this project's derived previews; open mappings are protected by Windows.
        for old in project.glob(f"{kind}_mesh-v*.npy"):
            if old != target:
                try:
                    old.unlink()
                except OSError:
                    pass
    return asset


def reference_vertices(packed, frame, count):
    """Independent NumPy reference for GPU verification; not used during playback."""
    cursor = count * 3
    template = packed[:cursor].reshape(count,3)
    shapes = packed[cursor:cursor+count*3*20].reshape(20,count*3)
    cursor += count*3*20
    poses = packed[cursor:cursor+count*3*486].reshape(486,count*3)
    cursor += count*3*486
    weights = packed[cursor:].reshape(count,55)
    posed = template + (frame[1366:1386] @ shapes + frame[880:1366] @ poses).reshape(count,3)
    transforms = (weights @ frame[:880].reshape(55,16)).reshape(count,4,4)
    return np.einsum("vij,vj->vi",transforms[:,:3,:3],posed) + transforms[:,:3,3] + frame[-3:]
