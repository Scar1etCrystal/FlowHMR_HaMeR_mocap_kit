import argparse
import os, sys, subprocess as _subproc
import pickle
from pathlib import Path

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
            os.environ['CUDA_VISIBLE_DEVICES'] = ''
            print(f"GPU incompatible (VRAM={_vram}MB CC={_cc}), forcing CPU mode")
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        print("GPU check crashed, forcing CPU mode")
except Exception:
    pass

import cv2
import numpy as np
import torch
from scipy.interpolate import interp1d
from scipy.signal import medfilt, savgol_filter
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

from hamer.models import DEFAULT_CHECKPOINT, load_hamer
from hamer.datasets.utils import convert_cvimg_to_tensor, expand_to_aspect_ratio, generate_image_patch_cv2
from hamer.utils.renderer import Renderer, cam_crop_to_full
from vitpose_model import ViTPoseModel


DEFAULT_VITPOSE_CKPT = Path("_DATA/vitpose_ckpts/vitpose+_huge/wholebody.pth")
LEFT_MIRROR = np.diag([-1.0, 1.0, 1.0]).astype(np.float32)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", type=str, required=True, help="Input video path")
    parser.add_argument("--out_folder", type=str, default="demo_hamer_out", help="Output folder")
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--detect_interval", type=int, default=2)
    parser.add_argument("--window", type=int, default=11, help="Savgol smoothing window, must be odd")
    parser.add_argument("--smplx_model_path", type=str, default="", help="Optional SMPL-X npz/pkl path for hand mean export")
    parser.add_argument("--skip_render", action="store_true", help="Export pkl only, skip overlay video")
    return parser.parse_args()


def build_detector(device):
    from detectron2 import model_zoo
    from hamer.utils.utils_detectron2 import DefaultPredictor_Lazy
    import hamer as _hamer_pkg

    cfg = model_zoo.get_config("new_baselines/mask_rcnn_regnety_4gf_dds_FPN_400ep_LSJ.py", trained=True)
    # Override with local model if available (portable pack)
    _regnety_local = Path(_hamer_pkg.__file__).parent / '..' / 'models' / 'model_final_ef3a80.pkl'
    if _regnety_local.exists():
        cfg.train.init_checkpoint = str(_regnety_local.resolve())
    cfg.model.roi_heads.box_predictor.test_score_thresh = 0.5
    return DefaultPredictor_Lazy(cfg, device=device)


def process_patch(cvimg, box, is_right, cfg):
    img_size = cfg.MODEL.IMAGE_SIZE
    mean = 255.0 * np.array(cfg.MODEL.IMAGE_MEAN)
    std = 255.0 * np.array(cfg.MODEL.IMAGE_STD)

    center = np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])
    scale = 2.0 * np.array([(box[2] - box[0]) / 200.0, (box[3] - box[1]) / 200.0])
    bbox_size = expand_to_aspect_ratio(scale * 200, target_aspect_ratio=cfg.MODEL.get("BBOX_SHAPE")).max()

    flip = is_right == 0
    img_patch_cv, _ = generate_image_patch_cv2(
        cvimg, center[0], center[1], bbox_size, bbox_size, img_size, img_size, flip, 1.0, 0, border_mode=cv2.BORDER_CONSTANT
    )
    img_patch = convert_cvimg_to_tensor(img_patch_cv[:, :, ::-1])
    for c in range(3):
        img_patch[c] = (img_patch[c] - mean[c]) / std[c]

    return {
        "img": img_patch,
        "box_center": center.astype(np.float32),
        "box_size": np.float32(bbox_size),
        "img_size": np.array([cvimg.shape[1], cvimg.shape[0]], dtype=np.float32),
        "right": np.float32(is_right),
    }


def smooth_rotations(rot_data, window, poly=3):
    original_shape = rot_data.shape
    is_matrix = len(original_shape) >= 3 and original_shape[-1] == 3 and original_shape[-2] == 3

    if is_matrix:
        rot_flat = rot_data.reshape(original_shape[0], -1, 3, 3)
    else:
        rot_flat = rot_data.reshape(original_shape[0], -1, 3)

    num_frames, num_joints = rot_flat.shape[0], rot_flat.shape[1]
    smoothed = np.zeros_like(rot_flat)

    for joint_idx in range(num_joints):
        if is_matrix:
            quats = R.from_matrix(rot_flat[:, joint_idx]).as_quat()
        else:
            quats = R.from_rotvec(rot_flat[:, joint_idx]).as_quat()

        for frame_idx in range(1, num_frames):
            if np.dot(quats[frame_idx], quats[frame_idx - 1]) < 0:
                quats[frame_idx] = -quats[frame_idx]

        smoothed_quats = savgol_filter(quats, window, poly, axis=0)
        smoothed_quats /= np.linalg.norm(smoothed_quats, axis=1, keepdims=True) + 1e-8

        if is_matrix:
            smoothed[:, joint_idx] = R.from_quat(smoothed_quats).as_matrix()
        else:
            smoothed[:, joint_idx] = R.from_quat(smoothed_quats).as_rotvec()

    return smoothed.reshape(original_shape)


def project_to_rotation_matrix(rot):
    u, _, vh = np.linalg.svd(rot)
    rot_proj = u @ vh
    if np.linalg.det(rot_proj) < 0:
        u[:, -1] *= -1
        rot_proj = u @ vh
    return rot_proj.astype(np.float32)


def load_smplx_hand_means(path):
    if not path:
        return None, None
    p = Path(path)
    if p.suffix == ".npz":
        data = np.load(p, allow_pickle=True)
    else:
        with open(p, "rb") as f:
            data = pickle.load(f, encoding="latin1")
    left_mean = np.asarray(data["hands_meanl"], dtype=np.float32).reshape(15, 3)
    right_mean = np.asarray(data["hands_meanr"], dtype=np.float32).reshape(15, 3)
    return left_mean, right_mean


def residual_to_abs_mats(residual_mats, mean_rotvec):
    residual_rotvec = R.from_matrix(residual_mats).as_rotvec().astype(np.float32)
    abs_rotvec = residual_rotvec + mean_rotvec
    return R.from_rotvec(abs_rotvec).as_matrix().astype(np.float32)


def decode_single_rot(value):
    arr = np.asarray(value, dtype=np.float32)
    flat = arr.reshape(-1)
    if flat.size == 3:
        rotvec = flat.astype(np.float32)
        matrix = R.from_rotvec(rotvec).as_matrix().astype(np.float32)
        return rotvec, project_to_rotation_matrix(matrix)
    if flat.size == 9:
        matrix = project_to_rotation_matrix(flat.reshape(3, 3))
        rotvec = R.from_matrix(matrix).as_rotvec().astype(np.float32)
        return rotvec, matrix
    raise ValueError(f"Unsupported single rotation size: {flat.size}")


def decode_hand_pose(value):
    arr = np.asarray(value, dtype=np.float32)
    flat = arr.reshape(-1)
    if flat.size == 45:
        rotvecs = flat.reshape(15, 3).astype(np.float32)
        matrices = R.from_rotvec(rotvecs).as_matrix().astype(np.float32)
        matrices = np.stack([project_to_rotation_matrix(m) for m in matrices], axis=0)
        rotvecs = R.from_matrix(matrices).as_rotvec().astype(np.float32)
        return rotvecs, matrices
    if flat.size == 135:
        matrices = flat.reshape(15, 3, 3).astype(np.float32)
        matrices = np.stack([project_to_rotation_matrix(m) for m in matrices], axis=0)
        rotvecs = R.from_matrix(matrices).as_rotvec().astype(np.float32)
        return rotvecs, matrices
    raise ValueError(f"Unsupported hand pose size: {flat.size}")


def build_video_writer(path, fps, width, height):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if writer.isOpened():
        return writer
    raise RuntimeError(f"Failed to open video writer for: {path}")


def main():
    args = parse_args()
    # Auto-select device: use CUDA only if GPU has enough VRAM (>=4GB) and CC >=7.0
    if torch.cuda.is_available():
        try:
            gpu_name = torch.cuda.get_device_name(0)
            vram_mb = torch.cuda.get_device_properties(0).total_memory // (1024*1024)
            major, minor = torch.cuda.get_device_capability(0)
            cc = major + minor * 0.1
            if cc >= 7.0 and vram_mb >= 4000:
                device = torch.device("cuda")
                print(f"Using CUDA: {gpu_name} ({vram_mb}MB, CC {cc})")
            else:
                device = torch.device("cpu")
                print(f"Falling back to CPU: {gpu_name} ({vram_mb}MB, CC {cc}) - need >=4GB VRAM and CC >=7.0")
        except Exception as e:
            device = torch.device("cpu")
            print(f"Falling back to CPU (GPU check failed: {e})")
    else:
        device = torch.device("cpu")
        print("Using CPU (no CUDA available)")

    if device.type == "cpu" and args.batch_size > 1:
        print(f"CPU mode: reducing HAMER batch_size from {args.batch_size} to 1 to lower memory use")
        args.batch_size = 1

    print("Loading HaMeR, detector, and pose model...")
    model, model_cfg = load_hamer(args.checkpoint)
    model = model.to(device).eval()
    detector = build_detector(device)
    pose_model = ViTPoseModel(device)
    renderer = Renderer(model_cfg, faces=model.mano.faces)
    left_mean, right_mean = load_smplx_hand_means(args.smplx_model_path)

    video_path = Path(args.video_path)
    out_dir = Path(args.out_folder)
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 1e-6:
        fps = 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    focal_length = model_cfg.EXTRA.FOCAL_LENGTH / model_cfg.MODEL.IMAGE_SIZE * max(width, height)

    detected_boxes = {}
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    for frame_idx in tqdm(range(num_frames), desc="[1/4] Detect and track", unit="frame"):
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % args.detect_interval != 0:
            continue

        det_out = detector(frame)
        valid_idx = (det_out["instances"].pred_classes == 0) & (det_out["instances"].scores > 0.5)
        pred_bboxes = det_out["instances"].pred_boxes.tensor[valid_idx].cpu().numpy()
        pred_scores = det_out["instances"].scores[valid_idx].cpu().numpy()
        vit_input = [np.concatenate([pred_bboxes, pred_scores[:, None]], axis=1)] if len(pred_bboxes) > 0 else []
        vit_out = pose_model.predict_pose(frame[:, :, ::-1], vit_input) if vit_input else []

        hand_cands = []
        for vpose in vit_out:
            for is_right, kps in ((0, vpose["keypoints"][-42:-21]), (1, vpose["keypoints"][-21:])):
                if kps[:, 2].max() <= 0.5:
                    continue
                valid = kps[:, 2] > 0.5
                hand_cands.append(
                    {
                        "bbox": np.array(
                            [kps[valid, 0].min(), kps[valid, 1].min(), kps[valid, 0].max(), kps[valid, 1].max()],
                            dtype=np.float32,
                        ),
                        "is_right": is_right,
                    }
                )
        detected_boxes[frame_idx] = hand_cands

    all_tasks = []
    for hand_type in [0, 1]:
        frame_indices = [f for f in sorted(detected_boxes.keys()) if any(h["is_right"] == hand_type for h in detected_boxes[f])]
        if not frame_indices:
            continue
        bboxes = [next(h["bbox"] for h in detected_boxes[f] if h["is_right"] == hand_type) for f in frame_indices]
        interp_func = interp1d(frame_indices, bboxes, axis=0, kind="linear", fill_value="extrapolate")
        for frame_idx in range(min(frame_indices), max(frame_indices) + 1):
            all_tasks.append({"frame_idx": frame_idx, "bbox": interp_func(frame_idx), "is_right": hand_type, "results": {}})

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    batch_img, batch_meta, batch_task_refs = [], [], []

    def flush_batch():
        if not batch_img:
            return
        box_centers = np.stack([m["box_center"] for m in batch_meta], axis=0).astype(np.float32)
        box_sizes = np.array([m["box_size"] for m in batch_meta], dtype=np.float32)
        img_sizes = np.stack([m["img_size"] for m in batch_meta], axis=0).astype(np.float32)
        rights = np.array([m["right"] for m in batch_meta], dtype=np.float32)
        batch_data = {
            "img": torch.stack([torch.from_numpy(img) for img in batch_img]).float().to(device),
            "right": torch.from_numpy(rights).to(device),
            "box_center": torch.from_numpy(box_centers).to(device),
            "box_size": torch.from_numpy(box_sizes).to(device),
            "img_size": torch.from_numpy(img_sizes).to(device),
        }
        with torch.no_grad():
            out = model(batch_data)

        pred_cam = out["pred_cam"].clone()
        pred_cam[:, 1] *= 2 * batch_data["right"] - 1
        cam_t = cam_crop_to_full(
            pred_cam, batch_data["box_center"], batch_data["box_size"], batch_data["img_size"], focal_length
        ).cpu().numpy()

        for idx, task in enumerate(batch_task_refs):
            task["results"] = {
                "pose": out["pred_mano_params"]["hand_pose"][idx].cpu().numpy(),
                "orient": out["pred_mano_params"]["global_orient"][idx].cpu().numpy(),
                "betas": out["pred_mano_params"]["betas"][idx].cpu().numpy(),
                "cam_t": cam_t[idx],
            }
        batch_img.clear()
        batch_meta.clear()
        batch_task_refs.clear()

    for frame_idx in tqdm(range(num_frames), desc="[2/4] Infer 3D hand", unit="frame"):
        ret, frame = cap.read()
        if not ret:
            break
        current_tasks = [t for t in all_tasks if t["frame_idx"] == frame_idx]
        for task in current_tasks:
            meta = process_patch(frame, task["bbox"], task["is_right"], model_cfg)
            batch_img.append(meta["img"])
            batch_meta.append(meta)
            batch_task_refs.append(task)
            if len(batch_img) >= args.batch_size:
                flush_batch()
    flush_batch()

    print("[3/4] Smoothing...")
    for hand_type in [0, 1]:
        hand_tasks = sorted([t for t in all_tasks if t["is_right"] == hand_type], key=lambda x: x["frame_idx"])
        if len(hand_tasks) < args.window:
            continue
        poses = np.array([t["results"]["pose"] for t in hand_tasks])
        orients = np.array([t["results"]["orient"] for t in hand_tasks])
        cams = np.array([t["results"]["cam_t"] for t in hand_tasks])

        s_poses = smooth_rotations(poses, args.window)
        s_orients = smooth_rotations(orients, args.window)
        s_cams = np.zeros_like(cams)
        for dim in range(3):
            s_cams[:, dim] = medfilt(cams[:, dim], kernel_size=5)
        s_cams = savgol_filter(s_cams, args.window, 3, axis=0)

        for idx, task in enumerate(hand_tasks):
            task["results"]["pose"] = s_poses[idx]
            task["results"]["orient"] = s_orients[idx]
            task["results"]["cam_t"] = s_cams[idx]

    export_data = {
        "global_orient": np.zeros((num_frames, 2, 3, 3), dtype=np.float32),
        "hand_pose": np.zeros((num_frames, 2, 15, 3, 3), dtype=np.float32),
        "global_orient_valid": np.zeros((num_frames, 2), dtype=np.uint8),
        "hand_pose_abs_flat": np.zeros((num_frames, 2, 15, 3, 3), dtype=np.float32),
        "global_orient_abs_lr": np.zeros((num_frames, 2, 3, 3), dtype=np.float32),
    }

    frames_export_data = [{"frame_idx": i, "detections": []} for i in range(num_frames)]
    for task in all_tasks:
        frames_export_data[task["frame_idx"]]["detections"].append(task)

    for frame_idx in tqdm(range(num_frames), desc="[4/4] Build export data", unit="frame"):
        for det in frames_export_data[frame_idx]["detections"]:
            res = det["results"]
            if not res:
                continue

            _, orient_matrix = decode_single_rot(res["orient"])
            _, pose_matrices = decode_hand_pose(res["pose"])

            hand_idx = 0 if det["is_right"] == 0.0 else 1

            export_data["global_orient"][frame_idx, hand_idx] = orient_matrix
            export_data["hand_pose"][frame_idx, hand_idx] = pose_matrices
            export_data["global_orient_valid"][frame_idx, hand_idx] = 1

            if det["is_right"] == 1.0:
                export_data["global_orient_abs_lr"][frame_idx, hand_idx] = orient_matrix
                if right_mean is not None:
                    export_data["hand_pose_abs_flat"][frame_idx, hand_idx] = residual_to_abs_mats(pose_matrices, right_mean)
            else:
                left_global = LEFT_MIRROR @ orient_matrix @ LEFT_MIRROR
                export_data["global_orient_abs_lr"][frame_idx, hand_idx] = left_global
                if right_mean is not None:
                    right_abs = residual_to_abs_mats(pose_matrices, right_mean)
                    export_data["hand_pose_abs_flat"][frame_idx, hand_idx] = LEFT_MIRROR[None] @ right_abs @ LEFT_MIRROR

    pkl_path = out_dir / f"{video_path.stem}_hamer_data.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(export_data, f)

    print(f"Saved hand pkl: {pkl_path}")
    
    if args.skip_render:
        cap.release()
        return

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    out_video_path = out_dir / f"{video_path.stem}_hamer.mp4"
    writer = None
    try:
        writer = build_video_writer(out_video_path, fps, width, height)
        for frame_idx in tqdm(range(num_frames), desc="[5/5] Render video", unit="frame"):
            ret, frame_bgr = cap.read()
            if not ret:
                break

            all_v, all_t, all_r = [], [], []
            for det in frames_export_data[frame_idx]["detections"]:
                res = det["results"]
                if not res:
                    continue

                with torch.no_grad():
                    # HaMeR's MANO wrapper expects the same representation it
                    # produced. Keep rendering on the proven raw path.
                    m_out = model.mano(
                        global_orient=torch.tensor(res["orient"]).float().unsqueeze(0).to(device),
                        hand_pose=torch.tensor(res["pose"]).float().unsqueeze(0).to(device),
                        betas=torch.tensor(res["betas"]).float().unsqueeze(0).to(device),
                    )
                v = m_out.vertices[0].cpu().numpy()
                v[:, 0] *= 2 * float(det["is_right"]) - 1
                all_v.append(v)
                all_t.append(res["cam_t"])
                all_r.append(float(det["is_right"]))

            if all_v:
                img_size = np.array([width, height], dtype=np.float32)
                cam_view = renderer.render_rgba_multiple(
                    all_v, cam_t=all_t, render_res=img_size, is_right=all_r, focal_length=focal_length
                )
                input_img = np.concatenate([frame_bgr[:, :, ::-1] / 255.0, np.ones_like(frame_bgr[:, :, :1])], axis=2)
                overlay = input_img[:, :, :3] * (1 - cam_view[:, :, 3:]) + cam_view[:, :, :3] * cam_view[:, :, 3:]
                frame_bgr = np.clip(255 * overlay[:, :, ::-1], 0, 255).astype(np.uint8)
            writer.write(frame_bgr)
    finally:
        cap.release()
        if writer is not None:
            writer.release()

    print(f"Saved video: {out_video_path}")


if __name__ == "__main__":
    main()
