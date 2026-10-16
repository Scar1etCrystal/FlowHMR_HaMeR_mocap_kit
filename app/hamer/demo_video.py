"""
HAMER Video Processing Script
- Extracts frames from video
- Detects hands using ViTDet + ViTPose
- Runs HAMER to get hand pose parameters (MANO format)
- Saves per-frame hand data as numpy arrays

Usage:
    python demo_video.py --video path/to/video.mp4 --out_folder output/dir [--batch_size 8] [--rescale_factor 2.0]
    
Output:
    {out_folder}/hand_data.pkl containing:
        'left_hand_pose': (N, 45) axis-angle, MANO left hand pose
        'right_hand_pose': (N, 45) axis-angle, MANO right hand pose  
        'left_global_orient': (N, 3) axis-angle
        'right_global_orient': (N, 3) axis-angle
        'left_betas': (N, 10)
        'right_betas': (N, 10)
        'left_is_detected': (N,) bool - whether left hand was detected in frame
        'right_is_detected': (N,) bool - whether right hand was detected in frame
"""

from pathlib import Path
import sys, subprocess as _subproc

# Pre-check GPU before importing torch: hide GPU if incompatible (CC<7.0 or VRAM<4GB)
try:
    _gpu = _subproc.run([sys.executable, '-c',
        'import torch; p=torch.cuda.get_device_properties(0); '
        'print(f"{p.total_memory//(1024*1024)} {p.major} {p.minor}")'],
        capture_output=True, text=True, timeout=20)
    if _gpu.returncode == 0:
        _vram, _maj, _min = _gpu.stdout.strip().split()
        _vram, _cc = int(_vram), int(_maj) + int(_min)*0.1
        if _cc < 7.0 or _vram < 4000:
            import os; os.environ['CUDA_VISIBLE_DEVICES'] = ''
            print(f"GPU incompatible (VRAM={_vram}MB CC={_cc}), forcing CPU mode")
    else:
        import os; os.environ['CUDA_VISIBLE_DEVICES'] = ''
        print("GPU check crashed, forcing CPU mode")
except Exception:
    pass

import torch
import argparse
import os
import cv2
import numpy as np
from tqdm import tqdm

from hamer.configs import CACHE_DIR_HAMER
from hamer.models import HAMER, download_models, load_hamer, DEFAULT_CHECKPOINT
from hamer.utils import recursive_to
from hamer.datasets.vitdet_dataset import ViTDetDataset, DEFAULT_MEAN, DEFAULT_STD
from hamer.utils.renderer import Renderer, cam_crop_to_full

from vitpose_model import ViTPoseModel

import pickle


def extract_frames(video_path, output_dir=None):
    """Extract frames from video. If output_dir provided, save frames as images."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Video: {video_path}")
    print(f"  FPS={fps}, Frames={total_frames}, Size={width}x{height}")
    
    frames = []
    for i in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
        if output_dir:
            frame_path = os.path.join(output_dir, f"frame_{i:06d}.jpg")
            cv2.imwrite(frame_path, frame)
    cap.release()
    
    print(f"  Extracted {len(frames)} frames")
    return frames, fps


def main():
    parser = argparse.ArgumentParser(description='HaMeR video processing')
    parser.add_argument('--video', type=str, required=True, help='Input video path')
    parser.add_argument('--out_folder', type=str, default=None, help='Output folder')
    parser.add_argument('--checkpoint', type=str, default=DEFAULT_CHECKPOINT, help='Path to pretrained model checkpoint')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for HAMER inference')
    parser.add_argument('--rescale_factor', type=float, default=2.0, help='Factor for padding the bbox')
    parser.add_argument('--body_detector', type=str, default='regnety', choices=['vitdet', 'regnety'])
    parser.add_argument('--start_frame', type=int, default=0, help='Start frame index')
    parser.add_argument('--end_frame', type=int, default=-1, help='End frame index (-1 for all)')

    args = parser.parse_args()

    if args.out_folder is None:
        video_stem = Path(args.video).stem
        args.out_folder = f"outputs/{video_stem}_hamer"
    
    os.makedirs(args.out_folder, exist_ok=True)

    # Download and load checkpoints
    download_models(CACHE_DIR_HAMER)
    model, model_cfg = load_hamer(args.checkpoint)

    # Setup HaMeR model
    # Auto-select device: use CUDA only if GPU has enough VRAM (>=4GB) and CC >=7.0
    if torch.cuda.is_available():
        try:
            gpu_name = torch.cuda.get_device_name(0)
            vram_mb = torch.cuda.get_device_properties(0).total_memory // (1024*1024)
            major, minor = torch.cuda.get_device_capability(0)
            cc = major + minor * 0.1
            if cc >= 7.0 and vram_mb >= 4000:
                device = torch.device('cuda')
                print(f"Using CUDA: {gpu_name} ({vram_mb}MB, CC {cc})")
            else:
                device = torch.device('cpu')
                print(f"Falling back to CPU: {gpu_name} ({vram_mb}MB, CC {cc}) - need >=4GB VRAM and CC >=7.0")
        except Exception as e:
            device = torch.device('cpu')
            print(f"Falling back to CPU (GPU check failed: {e})")
    else:
        device = torch.device('cpu')
        print("Using CPU (no CUDA available)")
    model = model.to(device)
    model.eval()

    # Load detector
    from hamer.utils.utils_detectron2 import DefaultPredictor_Lazy
    if args.body_detector == 'vitdet':
        from detectron2.config import LazyConfig
        import hamer
        cfg_path = Path(hamer.__file__).parent / 'configs' / 'cascade_mask_rcnn_vitdet_h_75ep.py'
        detectron2_cfg = LazyConfig.load(str(cfg_path))
        # Use local model if available, fallback to remote URL
        _vitdet_local = Path(hamer.__file__).parent / '..' / 'models' / 'model_final_f05665.pkl'
        if _vitdet_local.exists():
            detectron2_cfg.train.init_checkpoint = str(_vitdet_local.resolve())
        else:
            detectron2_cfg.train.init_checkpoint = "https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl"
        for i in range(3):
            detectron2_cfg.model.roi_heads.box_predictors[i].test_score_thresh = 0.25
        detector = DefaultPredictor_Lazy(detectron2_cfg)
    elif args.body_detector == 'regnety':
        from detectron2 import model_zoo
        from detectron2.config import get_cfg
        import hamer as _hamer_pkg
        detectron2_cfg = model_zoo.get_config('new_baselines/mask_rcnn_regnety_4gf_dds_FPN_400ep_LSJ.py', trained=True)
        _regnety_local = Path(_hamer_pkg.__file__).parent / '..' / 'models' / 'model_final_ef3a80.pkl'
        if _regnety_local.exists():
            detectron2_cfg.train.init_checkpoint = str(_regnety_local.resolve())
        detectron2_cfg.model.roi_heads.box_predictor.test_score_thresh = 0.5
        detectron2_cfg.model.roi_heads.box_predictor.test_nms_thresh = 0.4
        detector = DefaultPredictor_Lazy(detectron2_cfg)

    # Keypoint detector
    cpm = ViTPoseModel(device)

    # Extract frames
    print("Extracting frames from video...")
    frames, fps = extract_frames(args.video)
    total_frames = len(frames)
    
    if args.end_frame < 0:
        args.end_frame = total_frames
    frames = frames[args.start_frame:args.end_frame]
    num_frames = len(frames)
    
    print(f"Processing {num_frames} frames (from {args.start_frame} to {args.start_frame + num_frames})")

    # Per-frame storage
    # MANO hand_pose: (15, 3) = 45 params per hand in axis-angle
    left_hand_poses = np.zeros((num_frames, 45), dtype=np.float32)
    right_hand_poses = np.zeros((num_frames, 45), dtype=np.float32)
    left_global_orients = np.zeros((num_frames, 3), dtype=np.float32)
    right_global_orients = np.zeros((num_frames, 3), dtype=np.float32)
    left_betas = np.zeros((num_frames, 10), dtype=np.float32)
    right_betas = np.zeros((num_frames, 10), dtype=np.float32)
    left_detected = np.zeros(num_frames, dtype=bool)
    right_detected = np.zeros(num_frames, dtype=bool)
    # 3D data for floating hand visualization
    left_cam_t = np.zeros((num_frames, 3), dtype=np.float32)
    right_cam_t = np.zeros((num_frames, 3), dtype=np.float32)
    left_keypoints_3d = np.zeros((num_frames, 21, 3), dtype=np.float32)
    right_keypoints_3d = np.zeros((num_frames, 21, 3), dtype=np.float32)
    left_vertices = np.zeros((num_frames, 778, 3), dtype=np.float32)
    right_vertices = np.zeros((num_frames, 778, 3), dtype=np.float32)

    # Process each frame
    scaled_focal_length = 5000.0  # default, will be overwritten during processing
    progress_file = open(os.path.join(args.out_folder, 'progress.log'), 'w')
    for frame_idx in range(num_frames):
        if frame_idx % 10 == 0:
            pct = 100.0 * frame_idx / num_frames
            msg = f"Processing frame {frame_idx}/{num_frames} ({pct:.1f}%)\n"
            print(msg, end='', flush=True)
            progress_file.write(msg)
            progress_file.flush()
        img_cv2 = frames[frame_idx]

        # Detect humans in image
        det_out = detector(img_cv2)
        img = img_cv2.copy()[:, :, ::-1]

        det_instances = det_out['instances']
        valid_idx = (det_instances.pred_classes == 0) & (det_instances.scores > 0.5)
        pred_bboxes = det_instances.pred_boxes.tensor[valid_idx].cpu().numpy()
        pred_scores = det_instances.scores[valid_idx].cpu().numpy()

        if len(pred_bboxes) == 0:
            continue

        # Detect human keypoints for each person
        vitposes_out = cpm.predict_pose(
            img,
            [np.concatenate([pred_bboxes, pred_scores[:, None]], axis=1)],
        )

        bboxes = []
        is_right_list = []

        # Use hands based on hand keypoint detections
        for vitposes in vitposes_out:
            left_hand_keyp = vitposes['keypoints'][-42:-21]
            right_hand_keyp = vitposes['keypoints'][-21:]

            # Left hand
            keyp = left_hand_keyp
            valid = keyp[:, 2] > 0.5
            if sum(valid) > 3:
                bbox = [keyp[valid, 0].min(), keyp[valid, 1].min(),
                        keyp[valid, 0].max(), keyp[valid, 1].max()]
                bboxes.append(bbox)
                is_right_list.append(0)

            # Right hand
            keyp = right_hand_keyp
            valid = keyp[:, 2] > 0.5
            if sum(valid) > 3:
                bbox = [keyp[valid, 0].min(), keyp[valid, 1].min(),
                        keyp[valid, 0].max(), keyp[valid, 1].max()]
                bboxes.append(bbox)
                is_right_list.append(1)

        if len(bboxes) == 0:
            continue

        boxes = np.stack(bboxes)
        right = np.stack(is_right_list)

        # Run HAMER reconstruction on all detected hands
        dataset = ViTDetDataset(model_cfg, img_cv2, boxes, right, rescale_factor=args.rescale_factor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

        for batch in dataloader:
            batch = recursive_to(batch, device)
            with torch.no_grad():
                out = model(batch)

            pred_mano_params = out['pred_mano_params']
            batch_size = batch['img'].shape[0]

            # Compute full camera translation (same as run_hamer_video_viz.py)
            multiplier = (2 * batch['right'] - 1)
            pred_cam = out['pred_cam']
            pred_cam[:, 1] = multiplier * pred_cam[:, 1]
            box_center = batch["box_center"].float()
            box_size = batch["box_size"].float()
            img_size = batch["img_size"].float()
            scaled_focal_length = model_cfg.EXTRA.FOCAL_LENGTH / model_cfg.MODEL.IMAGE_SIZE * img_size.max(dim=0).values.max().item()
            pred_cam_t_full = cam_crop_to_full(pred_cam, box_center, box_size, img_size, scaled_focal_length).detach().cpu().numpy()

            for n in range(batch_size):
                is_right_val = int(batch['right'][n].cpu().numpy())
                # hand_pose is in rotation matrix format (15, 3, 3), convert to axis-angle (15, 3)=45
                hand_pose_rotmat = pred_mano_params['hand_pose'][n].cpu().numpy()  # (15, 3, 3)
                global_orient_rotmat = pred_mano_params['global_orient'][n].cpu().numpy()  # (1, 3, 3)
                betas_val = pred_mano_params['betas'][n].cpu().numpy()  # (10,)

                # Convert rotation matrices to axis-angle
                from scipy.spatial.transform import Rotation
                hand_pose_aa = Rotation.from_matrix(hand_pose_rotmat.reshape(-1, 3, 3)).as_rotvec().reshape(-1)  # (45,)
                global_orient_aa = Rotation.from_matrix(global_orient_rotmat.reshape(-1, 3, 3)).as_rotvec().reshape(-1)  # (3,)

                # Get 3D keypoints and vertices in MANO space
                verts = out['pred_vertices'][n].detach().cpu().numpy()  # (778, 3)
                kps3d = out['pred_keypoints_3d'][n].detach().cpu().numpy()  # (21, 3)
                # Flip X for left hand (HAMER processes both as right, then flips)
                is_r = int(batch['right'][n].cpu().numpy())
                verts[:, 0] = (2 * is_r - 1) * verts[:, 0]
                kps3d[:, 0] = (2 * is_r - 1) * kps3d[:, 0]
                cam_t = pred_cam_t_full[n]

                if is_right_val == 1:
                    right_hand_poses[frame_idx] = hand_pose_aa
                    right_global_orients[frame_idx] = global_orient_aa
                    right_betas[frame_idx] = betas_val
                    right_detected[frame_idx] = True
                    right_cam_t[frame_idx] = cam_t
                    right_keypoints_3d[frame_idx] = kps3d
                    right_vertices[frame_idx] = verts
                else:
                    left_hand_poses[frame_idx] = hand_pose_aa
                    left_global_orients[frame_idx] = global_orient_aa
                    left_betas[frame_idx] = betas_val
                    left_detected[frame_idx] = True
                    left_cam_t[frame_idx] = cam_t
                    left_keypoints_3d[frame_idx] = kps3d
                    left_vertices[frame_idx] = verts

        # Free GPU memory incrementally
        if frame_idx % 50 == 0:
            torch.cuda.empty_cache()

    progress_file.close()

    # Save results
    hand_data = {
        'left_hand_pose': left_hand_poses,
        'right_hand_pose': right_hand_poses,
        'left_global_orient': left_global_orients,
        'right_global_orient': right_global_orients,
        'left_betas': left_betas,
        'right_betas': right_betas,
        'left_is_detected': left_detected,
        'right_is_detected': right_detected,
        'left_cam_t': left_cam_t,
        'right_cam_t': right_cam_t,
        'left_keypoints_3d': left_keypoints_3d,
        'right_keypoints_3d': right_keypoints_3d,
        'left_vertices': left_vertices,
        'right_vertices': right_vertices,
        'scaled_focal_length': scaled_focal_length,
        'fps': fps,
        'num_frames': num_frames,
        'start_frame': args.start_frame,
        'video_path': str(args.video),
    }

    output_path = os.path.join(args.out_folder, 'hand_data.pkl')
    with open(output_path, 'wb') as f:
        pickle.dump(hand_data, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"\nResults saved to: {output_path}")
    print(f"  Left hand detected in {left_detected.sum()}/{num_frames} frames")
    print(f"  Right hand detected in {right_detected.sum()}/{num_frames} frames")


if __name__ == '__main__':
    main()
