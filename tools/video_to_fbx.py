"""Drag-and-drop production entry: video -> SAM N60 + FlowHMR + HaMeR N2 -> FBX.
Stages run in separate processes to release GPU models, with content/config keyed caches.
Only the ordinary FlowHMR three-frame smoother is disabled; original IK is retained.
"""
from pathlib import Path
import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import time
import traceback

PACK = Path(__file__).resolve().parents[1]
REPO = PACK/'app/flowhmr'
TOOLS = PACK/'tools'
VERSION = 'video-fbx-n60-query-v2'
STAGES = ['prepare', 'camera', 'sam', 'generate', 'hamer', 'merge']
PRODUCTS = {
    'prepare': ['working_30fps.mp4', 'main_track.npz', 'video_info.json'],
    'camera': ['camera.npz', 'camera_info.json'],
    'sam': ['sam_mean_features.pt', 'sam_mean_components.npz', 'sam_run_info.json'],
    'generate': ['body_origin.npz', 'body_postprocess.npz', 'generation_info.json'],
    'hamer': ['hands/working_30fps_hamer_data.pkl', 'hands/run_info.json', 'hands/working_30fps_sampling.npz'],
    'merge': ['final.fbx', 'final.npz', 'validation.json'],
}
INPUTS = {
    'prepare': [],
    'camera': ['working_30fps.mp4', 'video_info.json'],
    'sam': ['working_30fps.mp4', 'main_track.npz', 'camera.npz'],
    'generate': ['sam_mean_features.pt', 'camera.npz'],
    'hamer': ['working_30fps.mp4', 'main_track.npz'],
    'merge': ['body_postprocess.npz', 'hands/working_30fps_hamer_data.pkl'],
}


def products_for(stage, cfg):
    """Expected stage outputs; per_frame SAM adds the dense hand cache."""
    names = list(PRODUCTS[stage])
    if stage == 'sam' and cfg.get('sam_hand_mode') == 'per_frame':
        names.append('sam_hand_dense.npz')
    return names

def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for data in iter(lambda: f.read(1024*1024), b''):
            h.update(data)
    return h.hexdigest()

def save_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix+'.part')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    os.replace(tmp, path)

def configure(c):
    env = {'SKIP_KEYPOINT_PROMPT':'1', 'IMG_SIZE':'512', 'LAYER_DTYPE':'fp32',
           'GPU_HAND_PREP':'1', 'MHR_USE_CUDA_GRAPH':'0', 'POSTPROCESS_SKIP_SMOOTH':'1',
           'USE_COMPILE':'0', 'USE_COMPILE_BACKBONE':'0', 'INTERM_FULLGRAPH':'0',
           'DECODER_COMPILE':'1', 'INTERM_COMPILE':'1', 'INTERM_SLIM':'1',
           'DINOV3_LOCAL_REPO':str(REPO/'third_party/dinov3_hub/facebookresearch_dinov3_main'),
           'HAMER_THREADS':str(c['threads']), 'PYTHONUTF8':'1', 'PYTHONIOENCODING':'utf-8'}
    for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:
        env[key] = str(c['threads'])
    os.environ.update(env)
    sys.path.insert(0, str(REPO))
    for d in ['sam-3d-body', 'vggt-omega', 'YOLOX']:
        sys.path.insert(0, str(REPO/'third_party'/d))
    os.chdir(REPO)

def stage_prepare(job, cfg):
    import cv2
    import numpy as np
    import torch
    from flowhmr.utils.runtime_video_processing import VideoProcessor, build_human_detector
    video = job/'working_30fps.mp4'
    if not video.is_file():
        import imageio_ffmpeg
        source = json.loads((job/'request.json').read_text(encoding='utf-8'))['source']
        temp = job/'working_30fps.part.mp4'
        cmd = [imageio_ffmpeg.get_ffmpeg_exe(), '-y','-v','error','-i',source,
               '-vf','fps=30','-an','-c:v','libx264','-preset','fast','-crf','22',
               '-pix_fmt','yuv420p','-movflags','+faststart',str(temp)]
        subprocess.run(cmd, check=True)
        os.replace(temp, video)
    meta = VideoProcessor.get_video_meta(str(video))
    if meta.total_frames < 8:
        raise ValueError('Input must contain at least 8 frames after 30fps conversion')
    frame_limit = cfg['max_generation_frames']
    if cfg['attention_backend'] == 'original':
        frame_limit = min(frame_limit, 2400)
    if meta.total_frames > frame_limit:
        raise ValueError(f'{meta.total_frames} frames exceed the current single-pass budget '
                         f'{frame_limit} for {cfg["attention_backend"]}. No frames were discarded. '
                         'Use a shorter input or the validated query_batched backend.')
    torch.set_num_threads(cfg['threads']); cv2.setNumThreads(cfg['threads'])
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is not available in the bundled Python environment')
    torch.cuda.reset_peak_memory_stats()
    proc = VideoProcessor(build_human_detector(torch.device('cuda')))
    tracks = proc.detect_and_track(str(video))
    index = VideoProcessor.select_best_track(tracks)
    if index < 0:
        raise RuntimeError('No human track found; check that a single full-body person is visible')
    track = tracks.tracks[index]
    boxes = VideoProcessor.interpolate_track(track, tracks.total_frames)
    boxes = VideoProcessor.smooth_bboxes(boxes, tracks.width, tracks.height)
    assert boxes.shape == (meta.total_frames, 4)
    detected = np.zeros(meta.total_frames, dtype=bool)
    detected[track.frames.astype(int)] = True
    # Full video timeline; endpoint hold outside observed track explicitly recorded.
    observed_range = [int(track.frames.min()), int(track.frames.max())]
    np.savez(job/'main_track.npz', bbox=boxes.astype(np.float32),
             bbox_xyxy=boxes.astype(np.float32), detected=detected,
             start_end=np.array([0, meta.total_frames-1]), observed_range=observed_range)
    n = meta.total_frames
    iv = cfg['vggt_interval']
    if iv == 'auto':
        iv = max(10, math.ceil((n-1)/(cfg['vggt_max_samples']-1)))
    iv = int(iv)
    samples = list(range(0, n, iv))
    if samples[-1] != n-1:
        samples.append(n-1)
    info = {'frames':n, 'fps':30, 'width':meta.width, 'height':meta.height,
            'observed_track_range_0based':observed_range,
            'bbox_endpoint_holds_outside_observation':True, 'output_covers_all_source_frames':True,
            'vggt_interval':iv, 'vggt_samples':len(samples),
            'effective_frame_limit':frame_limit,
            'peak_alloc_GiB':torch.cuda.max_memory_allocated()/2**30,
            'peak_reserved_GiB':torch.cuda.max_memory_reserved()/2**30}
    save_json(job/'video_info.json', info)
    print('PREPARE_DONE', json.dumps(info), flush=True)

def stage_camera(job, cfg):
    import torch
    from flowhmr.utils.runtime_vggt_camera import extract_vggt_camera, release_vggt_model
    v = json.loads((job/'video_info.json').read_text(encoding='utf-8'))
    torch.cuda.reset_peak_memory_stats()
    extract_vggt_camera(str(job/'working_30fps.mp4'), output_npz=str(job/'camera.npz'),
                        frame_range=(0,v['frames']-1), expected_len=v['frames'],
                        device='cuda',frame_interval=v['vggt_interval'])
    save_json(job/'camera_info.json', {'frames':v['frames'], 'interval':v['vggt_interval'],
             'samples':v['vggt_samples'], 'peak_alloc_GiB':torch.cuda.max_memory_allocated()/2**30,
             'peak_reserved_GiB':torch.cuda.max_memory_reserved()/2**30,
             'other_pipeline_models_resident':False})
    release_vggt_model()

def stage_sam(job, cfg):
    sys.path.insert(0, str(TOOLS))
    from sam_mean_features import extract
    extract(job/'working_30fps.mp4', job/'main_track.npz', job/'camera.npz', job,
            cfg['sam_hand_interval'], cfg['sam_sampling_seed'],
            hand_mode=cfg['sam_hand_mode'])

def stage_generate(job, cfg):
    import numpy as np
    import torch
    from flowhmr.pipeline.v2m_inference import V2MInferenceEngine, build_feature_dict, _rotate_smplh_params_y180
    from flowhmr.utils.runtime_vggt_camera import _vggt_opencv_to_yup_rt
    from flowhmr.core.math.geometry import rot6d_to_rotation_matrix, rotation_matrix_to_angle_axis
    torch.set_num_threads(cfg['threads'])
    engine = V2MInferenceEngine.build_lite(
        str(REPO/'checkpoints/flowhmr_latest/flowhmr_latest.ckpt'),
        str(REPO/'configs/base/model_basic05B.yml'), device='cuda',
        validation_steps=cfg['flow_steps'],postprocess=True)
    features = torch.load(job/'sam_mean_features.pt',map_location='cpu',weights_only=True)
    with np.load(job/'camera.npz',allow_pickle=False) as z:
        camera = _vggt_opencv_to_yup_rt(torch.from_numpy(z['RT_vggt'].astype(np.float32)))
    n = len(features)
    assert features.shape == (n,3072) and len(camera)==n
    f = build_feature_dict(features,camera,engine.device)
    from contextlib import nullcontext
    import flowhmr.pipeline.pipeline_v2m as pipeline_module
    sys.path.insert(0, str(TOOLS))
    from attention_query_batch_experiment import install, network_forward, individual_forward

    backend = cfg['attention_backend']
    batch_context = (
        install(engine.pipeline.motion_transformer, local=True, refiner=True,
                chunk=cfg['attention_query_chunk'])
        if backend == 'query_batched' else nullcontext()
    )
    noise_digests = []
    original_randn = pipeline_module.randn_tensor
    def recorded_noise(*args, **kwargs):
        noise = original_randn(*args, **kwargs)
        digest = hashlib.sha256(noise.detach().float().cpu().numpy().tobytes()).hexdigest()
        noise_digests.append({'shape':list(noise.shape), 'sha256':digest})
        return noise

    print(f'[attention] {backend}, query chunk={cfg["attention_query_chunk"]}, '
          'refiner visibility=GLOBAL', flush=True)
    pipeline_module.randn_tensor = recorded_noise
    torch.cuda.reset_peak_memory_stats(); t = time.perf_counter()
    try:
        with batch_context, torch.no_grad():
            if backend == 'query_batched':
                net = engine.pipeline.motion_transformer
                assert net.forward.__func__ is network_forward
                assert net.text_refiner.individual_token_refiner.forward.__func__ is individual_forward
            output = engine.pipeline.generate(feature=f,seeds=[cfg['flow_seed']],length=n,
                                               cfg_scale=cfg['flow_cfg_scale'])
    finally:
        pipeline_module.randn_tensor = original_randn
    torch.cuda.synchronize();gen_seconds = time.perf_counter()-t
    gen_alloc = torch.cuda.max_memory_allocated()/2**30
    gen_reserved = torch.cuda.max_memory_reserved()/2**30
    # Snapshot the raw network output first: postprocess mutates these tensors in place.
    # Mirrors upstream FlowHMR, which writes <name>_origin.* before postprocessing and
    # <name>.* after, so downstream work can pick either stage.
    origin_raw = {'rot6d': output['rot6d'][0, :n].detach().cpu().clone(),
                  'trans': output['trans'][0, :n].detach().cpu().clone(),
                  'shapes': output['shapes'][0, :1].detach().cpu().clone()}
    if 'foot_contacts' in output:
        origin_raw['foot_contacts'] = output['foot_contacts'][0, :n].detach().cpu().clone()
    torch.cuda.reset_peak_memory_stats(); t = time.perf_counter()
    output = engine.postprocess.run(output)
    torch.cuda.synchronize();post_seconds = time.perf_counter()-t
    post_alloc = torch.cuda.max_memory_allocated()/2**30
    post_reserved = torch.cuda.max_memory_reserved()/2**30
    def build_params(rot6d, trans_, shapes, contacts):
        """Same fields and Y-180 convention for the raw and postprocessed files.

        Accepts CUDA or CPU tensors: the raw snapshot is already on CPU while the
        postprocessed tensors are still on the device.
        """
        rot6d = rot6d.detach().cpu()
        trans_ = trans_.detach().cpu()
        shapes = shapes.detach().cpu()
        p = rotation_matrix_to_angle_axis(rot6d_to_rotation_matrix(rot6d)).numpy().copy()
        t_ = trans_.numpy().copy()
        _rotate_smplh_params_y180(p, t_)
        out = dict(poses=p.astype(np.float32), trans=t_.astype(np.float32),
                   betas=shapes.numpy(), Rh=p[:, 0].copy(),
                   num_frames=n, mocap_framerate=30, gender='neutral')
        if contacts is not None:
            c = contacts.detach().cpu().numpy()
            assert c.shape == (n, 4), c.shape
            if not np.isfinite(c).all():
                raise RuntimeError('Non-finite foot_contacts in model output')
            out['foot_contacts'] = c.astype(np.float32)
        return out

    # Raw network output before postprocessing. Mirrors upstream FlowHMR, which
    # writes <name>_origin.* first and <name>.* after, so downstream work can
    # choose either stage. Field set and Y-180 convention are identical.
    origin_params = build_params(origin_raw['rot6d'], origin_raw['trans'],
                                 origin_raw['shapes'], origin_raw.get('foot_contacts'))
    np.savez(job/'body_origin.npz', **origin_params)

    # Postprocessed result; the later merge stage adds the HaMeR fingers.
    # Per-frame foot contact scores are (T, 4) float32 ordered
    # ['L_Ankle','L_Foot','R_Ankle','R_Foot'] -- same field name and layout as the
    # upstream save path, so existing downstream consumers work unchanged.
    # They are probabilities and are unaffected by the Y-180 rotation.
    contact_columns = ['L_Ankle', 'L_Foot', 'R_Ankle', 'R_Foot'] \
        if 'foot_contacts' in origin_params else None
    data = build_params(output['rot6d'][0], output['trans'][0], output['shapes'][0, :1],
                        output['foot_contacts'][0, :n] if 'foot_contacts' in output else None)
    np.savez(job/'body_postprocess.npz',**data)
    save_json(job/'generation_info.json',{'frames':n,'generation_seconds':gen_seconds,
              'postprocess_seconds':post_seconds,'postprocess':True,
              'POSTPROCESS_SKIP_SMOOTH':os.environ['POSTPROCESS_SKIP_SMOOTH'],
              'ordinary_three_frame_smooth':False, 'steps':cfg['flow_steps'],
              'cfg_scale':cfg['flow_cfg_scale'],'seed':cfg['flow_seed'],
              'attention_backend':backend,
              'attention_query_chunk':cfg['attention_query_chunk'] if backend == 'query_batched' else None,
              'refiner_visibility':'global', 'initial_noise':noise_digests,
              'whole_sequence_generation':True,
              'generation_peak_alloc_GiB':gen_alloc,
              'generation_peak_reserved_GiB':gen_reserved,
              'postprocess_peak_alloc_GiB':post_alloc,
              'postprocess_peak_reserved_GiB':post_reserved,
              'foot_contacts_exported':contact_columns is not None,
              'foot_contacts_columns':contact_columns,
              'foot_contacts_semantics':'per-frame contact probability; postprocess uses >0.7 as contact',
              'raw_output_npz':'body_origin.npz (pre-postprocess, no HaMeR fingers)',
              'peak_alloc_GiB':max(gen_alloc,post_alloc),
              'peak_reserved_GiB':max(gen_reserved,post_reserved)})
    print('GENERATION_DONE', n, flush=True)

def stage_hamer(job, cfg):
    command = [sys.executable,'-u',str(TOOLS/'hamer_pose_sampled.py'),
               '--video',str(job/'working_30fps.mp4'),'--main_track',str(job/'main_track.npz'),
               '--out_dir',str(job/'hands'),'--detect_interval',str(cfg['hamer_interval']),
               '--batch_size',str(cfg['hamer_batch_size']),'--window',str(cfg['hamer_smooth_window']),
               '--max_interp_gap',str(cfg['hamer_max_interp_gap'])]
    subprocess.run(command,check=True)

def validate_fbx(job):
    import fbx
    import numpy as np
    from scipy.spatial.transform import Rotation
    from flowhmr.core.export.fbx_utils import SMPLH_JOINT2NUM
    with np.load(job/'final.npz',allow_pickle=False) as z:
        poses=z['poses'];trans=z['trans'];n=len(poses)
    with np.load(job/'body_postprocess.npz',allow_pickle=False) as z:
        assert np.array_equal(poses[:,:22],z['poses'][:,:22]), 'Body/wrists changed during merge'
        assert np.array_equal(trans,z['trans']), 'Root changed during merge'
    manager=fbx.FbxManager.Create();manager.SetIOSettings(fbx.FbxIOSettings.Create(manager,fbx.IOSROOT))
    importer=fbx.FbxImporter.Create(manager,'')
    if not importer.Initialize(str(job/'final.fbx'),-1,manager.GetIOSettings()):
        raise RuntimeError(importer.GetStatus().GetErrorString())
    scene=fbx.FbxScene.Create(manager,'validate');assert importer.Import(scene);importer.Destroy()
    bones={};meshes=[]
    def walk(node):
        attr=node.GetNodeAttribute()
        if attr:
            typ=attr.GetAttributeType()
            if typ==fbx.FbxNodeAttribute.EType.eSkeleton: bones[node.GetName()]=node
            if typ==fbx.FbxNodeAttribute.EType.eMesh: meshes.append(node)
        for i in range(node.GetChildCount()):walk(node.GetChild(i))
    walk(scene.GetRootNode());assert len(bones)==52 and len(meshes)==1
    stack=scene.GetSrcObject(fbx.FbxCriteria.ObjectType(fbx.FbxAnimStack.ClassId),0)
    layer=stack.GetSrcObject(fbx.FbxCriteria.ObjectType(fbx.FbxAnimLayer.ClassId),0)
    euler=np.empty((n,52,3))
    for name,j in SMPLH_JOINT2NUM.items():
        for a,axis in enumerate('XYZ'):
            curve=bones[name].LclRotation.GetCurve(layer,axis)
            assert curve.KeyGetCount()==n
            assert abs(curve.KeyGetTime(n-1).GetSecondDouble()-(n-1)/30)<1e-7
            euler[:,j,a]=[curve.KeyGetValue(i) for i in range(n)]
    err=np.degrees((Rotation.from_euler('xyz',euler.reshape(-1,3),degrees=True).inv()*
                    Rotation.from_rotvec(poses.reshape(-1,3))).magnitude()).max()
    assert err<.005,err
    root=np.stack([[bones['Pelvis'].LclTranslation.GetCurve(layer,a).KeyGetValue(i)
                    for i in range(n)] for a in 'XYZ'],axis=-1)
    terr=float(np.max(np.abs((root-root[0])-(trans-trans[0])*100)))
    assert terr<.01,terr
    manager.Destroy()
    save_json(job/'validation.json',{'frames':n,'fps':30,'skeletons':52,'meshes':1,
               'body_root_wrists_preserved':True,'rotation_key_max_error_deg':float(err),
               'root_trajectory_key_max_error_cm':terr,'fbx_bytes':(job/'final.fbx').stat().st_size,
               'sdk_reimport':True})

def stage_merge(job, cfg):
    command=[sys.executable,str(TOOLS/'merge_hamer_smplh.py'),
             '--body',str(job/'body_postprocess.npz'),
             '--hands',str(job/'hands/working_30fps_hamer_data.pkl'),
             '--output',str(job/'final.fbx')]
    subprocess.run(command,check=True)
    validate_fbx(job)

HANDLERS = dict(prepare=stage_prepare,camera=stage_camera,sam=stage_sam,
                generate=stage_generate,hamer=stage_hamer,merge=stage_merge)

def checked_settings(path):
    c=json.loads(Path(path).read_text(encoding='utf-8'))
    c.setdefault('attention_backend', 'query_batched')
    c.setdefault('attention_query_chunk', 256)
    c.setdefault('sam_hand_mode', 'sample_linear')
    if c['sam_hand_mode'] not in ('sample_linear', 'mean', 'per_frame'):
        raise ValueError("sam_hand_mode must be 'sample_linear', 'mean' or 'per_frame'")
    if c['attention_backend'] not in ('query_batched', 'original'):
        raise ValueError('attention_backend must be query_batched or original')
    if type(c['attention_query_chunk']) is not int or c['attention_query_chunk'] < 1:
        raise ValueError('attention_query_chunk must be a positive integer')
    for key in ['sam_hand_interval','hamer_interval','hamer_batch_size','flow_steps','threads','max_generation_frames']:
        if not isinstance(c[key],int) or c[key]<1:raise ValueError(f'{key} must be a positive integer')
    if c['hamer_smooth_window']<5 or c['hamer_smooth_window']%2==0:
        raise ValueError('hamer_smooth_window must be odd and at least 5')
    if c['vggt_max_samples']<2:raise ValueError('vggt_max_samples must be >=2')
    if c['vggt_interval']!='auto' and int(c['vggt_interval'])<1:raise ValueError('vggt_interval must be auto or positive')
    if c['flow_cfg_scale']!=1.0:raise ValueError('This 12GiB preset is validated at flow_cfg_scale=1.0')
    return c

def signature(c):
    files=[Path(__file__),TOOLS/'attention_query_batch_experiment.py',
           TOOLS/'sam_mean_features.py',TOOLS/'hamer_pose_sampled.py',
           TOOLS/'merge_hamer_smplh.py',PACK/'app/hamer/hamer_locked.py',PACK/'app/hamer/vitpose_model.py',
           REPO/'flowhmr/pipeline/v2m_inference.py',REPO/'flowhmr/utils/runtime_video_processing.py',
           REPO/'flowhmr/utils/runtime_v2m_postprocess_simple.py',
           REPO/'flowhmr/utils/runtime_vggt_camera.py',REPO/'flowhmr/utils/runtime_sam_features.py',
           REPO/'third_party/sam-3d-body/sam_3d_body/models/meta_arch/sam3d_body.py',
           REPO/'flowhmr/pipeline/pipeline_v2m.py',
           REPO/'flowhmr/network/token_refiner.py',
           REPO/'flowhmr/network/hymotion_mmdit_for_v2m.py',
           REPO/'flowhmr/network/mmdit/attention.py',REPO/'configs/base/model_basic05B.yml']
    code={str(f.relative_to(PACK)):sha(f) for f in files}
    model_files=[REPO/'checkpoints/flowhmr_latest/flowhmr_latest.ckpt',
                 REPO/'ckpts/sam-3d-body-dinov3/model.ckpt',REPO/'ckpts/sam-3d-body-dinov3/assets/mhr_model.pt',
                 REPO/'ckpts/vggt-omega/vggt_omega_1b_512.pt',REPO/'ckpts/yolox/yolox_l.pth',
                 PACK/'app/hamer/_DATA/hamer_ckpts/checkpoints/hamer.ckpt',
                 PACK/'app/hamer/_DATA/vitpose_ckpts/vitpose+_huge/wholebody.pth']
    models={str(f.relative_to(PACK)):[f.stat().st_size,f.stat().st_mtime_ns] for f in model_files}
    return {'version':VERSION,'settings':c,'code_sha256':code,'model_size_mtime':models,
            'enforced':{'SKIP_KEYPOINT_PROMPT':1,'POSTPROCESS_SKIP_SMOOTH':1,
                        'sam_mode':'sampled_mean','hamer_tta':1}}

def run_input(source, root, c, fresh):
    source=Path(source).expanduser().resolve()
    if not source.is_file():raise FileNotFoundError(source)
    print(f'\nInput: {source}',flush=True)
    fingerprint={'source_sha256':sha(source),**signature(c)}
    identity=hashlib.sha256(json.dumps(fingerprint,sort_keys=True).encode()).hexdigest()[:12]
    name=re.sub(r'[^\w.\-]+','_',source.stem,flags=re.UNICODE)[:65] or 'video'
    job=root/f'{name}_{identity}'
    if fresh:job=job.with_name(job.name+'_'+time.strftime('%Y%m%d_%H%M%S'))
    job.mkdir(parents=True,exist_ok=True)
    lock=job/'.running'
    try:
        fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    except FileExistsError:
        raise RuntimeError(f'This job is already running or was interrupted. '
                           f'Use --fresh for a separate run. Lock: {lock}')
    os.close(fd)
    request={'source':str(source),'fingerprint':fingerprint,'created':time.strftime('%Y-%m-%d %H:%M:%S')}
    save_json(job/'request.json',request);save_json(job/'settings.json',c)
    manifest={'source':str(source),'job':str(job),'settings':c,'status':'running','stages':{}}
    save_json(job/'run_manifest.json',manifest)
    begin=time.perf_counter()
    mode = c['sam_hand_mode']
    mode_text = {'sample_linear': f"SAM hands every {c['sam_hand_interval']} frames + linear interpolation",
                 'mean': f"SAM hands every {c['sam_hand_interval']} frames + whole-clip mean",
                 'per_frame': 'SAM hands every frame'}[mode]
    print(f'Output: {job}\n{mode_text}, HaMeR N={c["hamer_interval"]}, '
          'Fast + SKIP_KEYPOINT_PROMPT=1, postprocess on / ordinary smooth off',flush=True)
    print(f'Attention: {c["attention_backend"]}, query chunk={c["attention_query_chunk"]}, '
          'global refiner preserved',flush=True)
    try:
        for stage in STAGES:
            marker=job/f'{stage}.done.json';cached=False
            input_hashes={p:sha(job/p) for p in INPUTS[stage]}
            if marker.is_file():
                old=json.loads(marker.read_text(encoding='utf-8'))
                cached=all((job/p).is_file() and sha(job/p)==h for p,h in old.get('products',{}).items())
                cached=cached and set(old.get('products',{}))==set(products_for(stage,c))
                cached=cached and old.get('inputs')==input_hashes
            if cached:
                print(f'[{stage}] CACHE verified',flush=True)
                manifest['stages'][stage]={'cached':True,'seconds_this_run':0,'original_seconds':old['seconds']}
                continue
            print(f'\n=== {stage.upper()} ===',flush=True);t=time.perf_counter()
            command=[sys.executable,'-u',str(Path(__file__)), '--worker',stage,'--job',str(job)]
            with open(job/f'{stage}.log','w',encoding='utf-8') as log:
                proc=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                                      text=True,encoding='utf-8',errors='replace')
                try:
                    for line in proc.stdout:
                        log.write(line);log.flush();print(line,end='',flush=True)
                    rc=proc.wait()
                except BaseException:
                    proc.kill();proc.wait();raise
                if rc:raise RuntimeError(f'{stage} failed (exit {rc}). See {job/stage}.log')
            secs=time.perf_counter()-t
            for name in products_for(stage,c):
                if not (job/name).is_file():raise RuntimeError(f'{stage} output missing: {name}')
            save_json(marker,{'seconds':secs,'inputs':input_hashes,
                              'products':{p:sha(job/p) for p in products_for(stage,c)}})
            manifest['stages'][stage]={'cached':False,'seconds_this_run':secs}
            save_json(job/'run_manifest.json',manifest)
        manifest.update(status='completed',elapsed_seconds=time.perf_counter()-begin,
                        final_fbx=str(job/'final.fbx'),final_npz=str(job/'final.npz'))
        save_json(job/'run_manifest.json',manifest)
        print(f'\nDONE: {job/"final.fbx"}\nData: {job/"final.npz"}\nTime: {manifest["elapsed_seconds"]:.1f}s',flush=True)
        return job
    except BaseException as e:
        manifest.update(status='failed',error=str(e),elapsed_seconds=time.perf_counter()-begin)
        save_json(job/'run_manifest.json',manifest)
        raise
    finally:
        if lock.resolve().parent==job.resolve():lock.unlink(missing_ok=True)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('videos',nargs='*')
    ap.add_argument('--settings',default=str(PACK/'pipeline_settings.json'))
    ap.add_argument('--output-root',default=str(PACK/'outputs/video_to_fbx'))
    ap.add_argument('--fresh',action='store_true')
    ap.add_argument('--worker',choices=STAGES,help=argparse.SUPPRESS)
    ap.add_argument('--job',help=argparse.SUPPRESS)
    a=ap.parse_args()
    if a.worker:
        job=Path(a.job).resolve();c=checked_settings(job/'settings.json');configure(c)
        HANDLERS[a.worker](job,c)
        return 0
    if not a.videos:ap.error('Drop a video onto the BAT, or pass one or more video paths')
    c=checked_settings(a.settings)
    sources=[str(Path(v).expanduser().resolve()) for v in a.videos]
    root=Path(a.output_root).resolve()
    configure(c);root.mkdir(parents=True,exist_ok=True)
    failed=[]
    for video in sources:
        try:run_input(video,root,c,a.fresh)
        except Exception as e:
            traceback.print_exc();failed.append(str(video));print(f'FAILED: {video}: {e}',flush=True)
    if failed:
        print(f'Failed {len(failed)}/{len(a.videos)} input(s)',flush=True)
        return 1
    return 0

if __name__=='__main__':
    try:sys.exit(main())
    except Exception:
        traceback.print_exc();sys.exit(1)
