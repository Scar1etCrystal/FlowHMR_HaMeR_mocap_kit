"""Atomic native SMPL-H surface + anatomical Humanoid52 export preparation."""
from pathlib import Path
import json, os, shutil, uuid
import numpy as np
from .storage import atomic_json,sha256
from .result_artifacts import file_stat,begin_update,commit_artifacts


def finger_mapping():
    # Target finger order is Thumb/Index/Middle/Ring/Pinky; SMPL-H stores
    # Index/Middle/Pinky/Ring/Thumb. The body prefix is identical.
    return np.array([*range(22),*[j+side for side in (0,15) for first in (34,22,25,31,28) for j in range(first,first+3)]])


def target_rig(rest,vertices=None,skin=None):
    from .mixamo_rig import skeleton
    # skeleton() accepts SMPL-X anatomy ordering. Only this small joint view is
    # reordered; no surface/model conversion or approximate body fitting occurs.
    anatomy=np.zeros((55,3));anatomy[:22]=rest[:22];anatomy[25:55]=rest[22:52]
    rig=skeleton(rest[:22],anatomy,source_anchors=True)
    mapping=finger_mapping()
    # Incoming finger pivots are actor-specific native SMPL-H, including palms.
    for j in range(22,52):
        p=rig['parents'][j];sj=mapping[j];sp=mapping[p]
        rig['rest'][j]=rig['rest'][p]+rest[sj]-rest[sp]
    if vertices is not None:
        # Standard SMPL-H surface landmark indices (also documented by the
        # upstream smplx vertex selector). No registered model data is embedded.
        tips=((2746,2319,2445,2556,2673),(6191,5782,5905,6016,6133))
        for side in range(2):
            for finger,vertex in enumerate(tips[side]):
                j=22+side*15+finger*3+2
                rig['ends'][j]=vertices[vertex]-rest[mapping[j]]
        rig['ends'][10]=vertices[[3216,3226]].mean(0)-rest[10]
        rig['ends'][11]=vertices[[6617,6624]].mean(0)-rest[11]
        head=vertices[np.asarray(skin).argmax(1)==15]
        if not len(head):raise ValueError('SMPL-H head anatomy is unavailable')
        top=head[head[:,1].argmax()]
        rig['ends'][15]=top-rest[15]
    from .source_animation import source_rig
    return source_rig(rig,rest),mapping


def validate_axes(local,rig=None):
    from .mixamo_rig import HINGES
    maximum=0.
    for upper,middle,_,axis in HINGES:
        axis=np.asarray(axis,float)
        from .source_pose_fit import hinge_motion
        m=local[:,middle] if rig is None else hinge_motion(rig,local,upper,middle)
        error=np.linalg.norm(m@axis-axis,axis=-1)
        sine=np.stack((m[:,2,1]-m[:,1,2],m[:,0,2]-m[:,2,0],m[:,1,0]-m[:,0,1]),-1)@axis*.5
        angle=np.arctan2(sine,(np.trace(m,axis1=1,axis2=2)-1)*.5)
        maximum=max(maximum,float(error.max()))
        if maximum>1e-5 or angle.min() < -1e-5 or angle.max()>np.deg2rad(160 if upper<3 else 150)+1e-5:
            raise ValueError('An exported knee/elbow left its signed anatomical hinge')
    return maximum


def proper_rotations(matrices):
    """Remove only FP32 roundoff before long FK chains / BVH Euler encoding."""
    from scipy.spatial.transform import Rotation
    matrices=np.asarray(matrices,dtype=float)
    if not np.isfinite(matrices).all() or np.max(np.abs(matrices.swapaxes(-1,-2)@matrices-np.eye(3)))>2e-4 or np.min(np.linalg.det(matrices))<.9998:
        raise ValueError('FlowHMR produced invalid source rotation frames')
    return Rotation.from_matrix(matrices.reshape(-1,3,3)).as_matrix().reshape(matrices.shape)


def prepare(project,emit=lambda *a,**k:None,kind='initial'):
    import torch
    from flowhmr.core.bodymodels.smpl_skeleton import SMPLMesh,SMPLSkeleton
    from flowhmr.core.math.geometry import rot6d_to_rotation_matrix
    from .flow_backend import load_tensor,validate_motion
    from .models import BY_KEY
    from .mixamo_rig import VERSION,fk,mannequin,write_bvh
    from .source_animation import fit,temporal_halo,FIT_VERSION,validate,policy
    from .source_bvh import write_bvh
    from .bvh_bind import bind_pose,asset_frames
    from .mesh_data import adjacency
    from .mixamo_data import code_identity
    project=Path(project)
    from .motion_tools import result_is_current,RESULTS
    if kind not in ('initial','corrected_initial'):raise ValueError('Unsupported FlowHMR preview result')
    if not result_is_current(project,kind):raise ValueError('FlowHMR inference is incomplete or outdated. Run Reconstruct 3D before preparing its preview.')
    state=json.loads((project/'project.json').read_text(encoding='utf-8'));n=state['frames'];fps=float(state['fps'])
    filename=RESULTS[kind][0]
    params=load_tensor(project/filename);validate_motion(params,n)
    model=BY_KEY['smplh'].path;model_stamp=file_stat(model)
    source_stamp=file_stat(project/filename)
    begin_update(project,kind,'export')
    mesh=SMPLMesh(model_path=str(model));body=SMPLSkeleton(model_path=str(model))
    shape=params['shapes'].reshape(1,16)
    # Derive rest anatomy with the same upstream model, shape and zero pose.
    neutral=dict(rot_mats=torch.eye(3).reshape(1,1,3,3).expand(1,52,3,3),shapes=shape,trans=torch.zeros(1,3))
    with torch.no_grad():
        rest=body(neutral)['keypoints3d'][0].numpy()
        template=mesh(neutral)['vertices'][0].numpy()
    rig,mapping=target_rig(rest,template,mesh.lbs_weights.numpy());binding=bind_pose(rig)
    from .fbx_bind import prepare as prepare_fbx_bind
    fbx_binding=prepare_fbx_bind(rig,rest,template,'smplh',fps)
    dummy=mannequin(rig)
    token=uuid.uuid4().hex[:12];stem=f'{kind}_flow-{token}'
    files=dict(frames_file=stem+'-rig.npy',joints_file=stem+'-rig-joints.npy',reference_joints_file=stem+'-reference.npy',
               asset_file=stem+'-rig-asset.npz',bvh_file=stem+'.bvh',raw_file=stem+'-surface.npy',raw_asset=stem+'-surface-asset.npz')
    pending=project/(kind+'_mixamo_pending.json');atomic_json(pending,dict(generation=token))
    arrays=[];created=[];committed=False
    def mapped(name,shape):
        path=project/files[name];created.append(path)
        value=np.lib.format.open_memmap(path,mode='w+',dtype=np.float32,shape=shape);arrays.append(value);return value
    try:
        surface=mapped('raw_file',(n,6890,3));frames=mapped('frames_file',(n,52,4,4));points=mapped('joints_file',(n,22,3));reference=mapped('reference_joints_file',(n,22,3))
        roots=np.empty((n,3),np.float32);locals_out=np.empty((n,52,3,3),np.float32)
        native_points=np.empty((n,22,3),np.float32);native_rotations=np.empty((n,22,3,3),np.float32)
        errors=[];finger_errors=[];facing_errors=[];distal_error=0.;hinge_error=0.
        with torch.inference_mode():
            # Evaluate the selected original surface once, before direction
            # fitting. Large geometry stays in a bounded-batch disk mapping.
            for at in range(0,n,8):
                finish=min(n,at+8)
                vertices=mesh(dict(rot6d=params['rot6d'][0,at:finish],shapes=shape,trans=params['trans'][0,at:finish]))['vertices'].numpy()
                if not np.isfinite(vertices).all():raise ValueError('Non-finite SMPL-H surface')
                surface[at:finish]=vertices
                if at%128==0:emit('progress',current=finish,total=n,message='Prepare selected native SMPL-H surface')
            from .source_surface import native_endpoints
            surface_endpoints=native_endpoints(surface)
            for first in range(0,n,64):
                end=min(n,first+64);halo=temporal_halo(fps);begin=max(0,first-halo);stop=min(n,end+halo)
                selected=dict(rot6d=params['rot6d'][0,begin:stop],shapes=shape,trans=params['trans'][0,begin:stop])
                joints=body(selected)['keypoints3d'].numpy()
                local_native=proper_rotations(rot6d_to_rotation_matrix(selected['rot6d']).numpy())
                world_native=local_native.copy()
                for j,p in enumerate(body.parents.tolist()):
                    if p>=0:world_native[:,j]=world_native[:,p]@world_native[:,j]
                local,posed,world=fit(rig,joints[:,0],world_native[:,:22],joints[:,:22],
                    surface_endpoints=surface_endpoints[begin:stop],fps=fps)
                # Native hand animation is transferred in global rotation space;
                # each outgoing phalanx retains its own source orientation.
                for j in range(22,52):
                    p=rig['parents'][j]
                    world[:,j]=world_native[:,mapping[j]]
                    local[:,j]=world[:,p].swapaxes(-1,-2)@world[:,j]
                posed,world=fk(rig,joints[:,0],local)
                core=slice(first-begin,end-begin);local=local[core];world=world[core];posed=posed[core];joints=joints[core]
                native_rotations[first:end]=world_native[core,:22]
                native_points[first:end]=joints[:,:22];reference[first:end]=joints[:,:22]
                errors.append(np.linalg.norm(posed[:,:22]-joints[:,:22],axis=-1))
                finger_errors.append(np.linalg.norm(posed[:,22:]-joints[:,mapping[22:]],axis=-1))
                for j in (7,8,20,21):distal_error=max(distal_error,float(np.abs(world[:,j]-world_native[core,j]).max()))
                hinge_error=max(hinge_error,validate(local))
                from .source_pose_fit import anatomical_axes
                from .mixamo_rig import HINGES
                anatomical=anatomical_axes(rig,world)
                expected=np.stack([world_native[core,upper]@np.asarray(axis) for upper,_,_,axis in HINGES],axis=1)
                facing_errors.append(np.rad2deg(np.arccos(np.clip(np.sum(anatomical*expected,axis=-1),-1,1))))
                roots[first:end]=joints[:,0];locals_out[first:end]=local;points[first:end]=posed[:,:22]
                frames[first:end]=np.eye(4);frames[first:end,:,:3,:3]=world@binding['basis'];frames[first:end,:,:3,3]=posed
                emit('progress',current=end,total=n,message='Prepare native SMPL-H and export bones')
        if distal_error>1e-5:raise ValueError('Wrist or ankle orientation changed during FlowHMR retargeting')
        offsets,neighbors=adjacency(mesh.faces.astype(np.uint32),6890)
        np.savez(project/files['raw_asset'],faces=mesh.faces.astype(np.uint32),offsets=offsets,neighbors=neighbors,vertex_count=np.array(6890))
        np.savez(project/files['asset_file'],packed=asset_frames(dummy['packed'],binding),faces=dummy['faces'],vertex_count=np.array(dummy['vertex_count']),
                 rest=binding['rest'],parents=rig['parents'],ends=binding['ends'],export_basis=binding['basis'],anatomical_calibration=rig['alignment'],**fbx_binding)
        bvh_encoding=write_bvh(project/files['bvh_file'],rig,roots,locals_out,fps)
        created.extend(project/files[k] for k in ('raw_asset','asset_file','bvh_file'))
        for array in arrays:array.flush();array._mmap.close()
        arrays=[]
        np.savez_compressed(project/(kind+'_joints.npz'),joints=native_points,rotations=native_rotations,fps=np.array(fps))
        shutil.copy2(project/files['bvh_file'],project/(kind+'_motion.bvh'))
        atomic_json(project/(kind+'_mesh.json'),dict(version=1,representation='smplh',frames_file=files['raw_file'],asset=str(project/files['raw_asset']),
            source_size=source_stamp[0],source_mtime_ns=source_stamp[1],model_size=model_stamp[0],model_mtime_ns=model_stamp[1]))
        code=code_identity();code['flow_export.py']=sha256(Path(__file__))
        error=np.concatenate(errors);finger_error=np.concatenate(finger_errors)
        fit_summary=dict(mean_m=float(error.mean()),p95_m=float(np.quantile(error,.95)),max_m=float(error.max()),
            joints={name:dict(mean_m=float(error[:,j].mean()),max_m=float(error[:,j].max())) for j,name in enumerate(rig['names'][:22])})
        facing=np.concatenate(facing_errors)
        fit_summary['anatomical_facing_deg']={rig['names'][middle]:dict(mean=float(facing[:,i].mean()),p95=float(np.quantile(facing[:,i],.95)),max=float(facing[:,i].max())) for i,(_,middle,_,_) in enumerate(HINGES)}
        sources={filename:source_stamp,str(model):model_stamp,kind+'_joints.npz':file_stat(project/(kind+'_joints.npz'))}
        correction={}
        if kind.startswith('corrected_'):
            correction=json.loads((project/(kind+'_correction.json')).read_text(encoding='utf-8'))
            for suffix in ('correction.json','contacts.npz'):sources[kind+'_'+suffix]=file_stat(project/(kind+'_'+suffix))
        if file_stat(model)!=model_stamp or file_stat(project/filename)!=source_stamp or not result_is_current(project,kind):raise ValueError('FlowHMR source changed while exporting; retry.')
        atomic_json(project/(kind+'_mixamo.json'),dict(version=VERSION,kind=kind,frames=n,fps=fps,names=rig['names'],sources=sources,**files,
            raw_vertex_count=6890,representation='smplh',source_model='flowhmr',model_key='smplh',model_path=str(model),model_stat=model_stamp,code=code,
            code_stats={name:file_stat(ROOT/'mocap'/name) for name in code},selected_source_fit=fit_summary,source_fit=fit_summary,
            finger_fit_max_m=float(finger_error.max()),finger_fit_mean_m=float(finger_error.mean()),hinge_axis_max_error=hinge_error,
            **policy(fps),
            terminal_anatomy='Registered SMPL-H fingertips/toes and shaped head envelope; no generic third-phalanx lengths',
            distal_rotation_max_error=distal_error,correction=correction,geometry='Native SMPL-H / anatomical Humanoid52',
            export_bind=dict(version=binding['bind_version'],length_unit='meter',up_axis='Y',root_offset=binding['rest'][0].tolist(),
                rotation_order_default='Per joint',joint_rotation_orders=dict(zip(rig['names'],bvh_encoding['orders'])),minimum_singularity_margin=bvh_encoding['minimum_singularity_margin']),
            outputs={name:file_stat(project/name) for name in files.values()}))
        pending.unlink(missing_ok=True)
        commit_artifacts(project,kind,source_stamp,fps)
        shutil.copy2(project/files['bvh_file'],project/'motion.bvh')
        atomic_json(project/'export.json',dict(format='BVH',frames=n,fps=fps,result_kind=kind,skeleton='Humanoid52',length_unit='meter',up_axis='Y',rotation_order='Per joint',joints=rig['names'],
            source='FlowHMR native SMPL-H',finger_motion='Native FlowHMR animated hand rotations',bind_pose=dict(root_offset=binding['rest'][0].tolist())))
        committed=True
        # Only after a complete commit, collect superseded immutable generations.
        current=set(files.values())
        for path in project.glob(kind+'_flow-*'):
            if path.name not in current:
                try:path.unlink(missing_ok=True)
                except OSError:pass  # Old preview may still hold an immutable read handle.
        return project/'motion.bvh'
    finally:
        for array in arrays:
            try:array._mmap.close()
            except Exception:pass
        if not committed:
            # Unpublished generations have no readers. Keep pending markers,
            # but bound disk usage across failed/repeated preparations.
            for path in project.glob(stem+'*'):
                if path.is_file() and path.resolve().parent==project.resolve():
                    try:path.unlink()
                    except OSError:pass


from .paths import ROOT
