import argparse
import pickle
from pathlib import Path
import sys
import os

import cv2
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

import smplx

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


SMPLX_PARENTS_22 = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19]
LEFT_WRIST_BODY_SLICE = slice(57, 60)
RIGHT_WRIST_BODY_SLICE = slice(60, 63)
LEFT_HAND_INDEX = 0
RIGHT_HAND_INDEX = 1
LEFT_MIRROR = np.diag([-1.0, 1.0, 1.0]).astype(np.float32)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fuse GVHMR body motion with HaMeR hand motion, export SMPL-X pkl, and render an overlay check video."
    )
    parser.add_argument("--gvhmr_pkl", type=str, required=True, help="GVHMR output pkl")
    parser.add_argument(
        "--gvhmr_full_pkl",
        type=str,
        default="",
        help=(
            "Optional original GVHMR hmr4d_results.pt/pkl containing both smpl_params_incam and "
            "smpl_params_global. This is strongly preferred over flat exported person pkls."
        ),
    )
    parser.add_argument("--hamer_pkl", type=str, required=True, help="HaMeR output pkl")
    parser.add_argument("--video_path", type=str, help="Original video path for overlay rendering")
    parser.add_argument("--out_pkl", type=str, default="final_merged_smplx.pkl", help="Merged SMPL-X output pkl")
    parser.add_argument("--smplx_model_dir", type=str, required=True, help="Directory containing SMPL-X models")
    parser.add_argument("--gender", type=str, default="neutral", choices=["neutral", "male", "female"])
    parser.add_argument("--model_ext", type=str, default="npz", choices=["npz", "pkl"])
    parser.add_argument("--device", type=str, default="auto", help="auto/cuda/cpu")
    parser.add_argument(
        "--export_branch",
        type=str,
        default="auto",
        choices=["auto", "incam", "global", "flat"],
        help="Which merged branch is written to the pkl top level. Auto prefers global for Blender.",
    )
    parser.add_argument(
        "--render_branch",
        type=str,
        default="auto",
        choices=["auto", "incam", "global", "flat"],
        help="Which merged branch is used for overlay rendering. Auto prefers incam for video overlay.",
    )
    parser.add_argument(
        "--wrist_reference_branch",
        type=str,
        default="auto",
        choices=["auto", "incam", "global", "flat"],
        help=(
            "GVHMR branch used to convert HaMeR camera-space wrist global rotations into SMPL-X local wrist rotations. "
            "Auto prefers incam because HaMeR global_orient is full-image camera-space."
        ),
    )
    parser.add_argument(
        "--no_auto_full_gvhmr",
        action="store_true",
        help="Disable automatic lookup of outputs/demo/<video_stem>/hmr4d_results.pt when --gvhmr_pkl is flat.",
    )
    parser.add_argument("--render_max_frames", type=int, default=-1, help="Limit rendered frames for quick checks")
    parser.add_argument("--skip_render", action="store_true", help="Only export pkl, skip overlay rendering")
    parser.add_argument(
        "--disable_left_hand_mirror",
        action="store_true",
        help="Disable the standard left-hand mirror conversion for HaMeR->SMPL-X",
    )
    parser.add_argument(
        "--keep_residual_hand_pose",
        action="store_true",
        help="Deprecated compatibility flag. The default export already follows mano2smplx and keeps HaMeR hand rotations direct.",
    )
    parser.add_argument(
        "--bake_hand_mean",
        action="store_true",
        help="Experimental: add SMPL-X hand mean to HaMeR hand rotations. Disabled by default because mano2smplx does not do this.",
    )
    parser.add_argument(
        "--root_motion_mode",
        type=str,
        default="original",
        choices=["original", "inplace_xz", "inplace_xyz"],
        help=(
            "Post-process exported root translation. Use inplace_xz for dancing-in-place clips: "
            "it removes horizontal X/Z drift while keeping vertical motion."
        ),
    )
    parser.add_argument(
        "--wrist_twist_mode",
        type=str,
        default="none",
        choices=["none", "soft_clamp"],
        help=(
            "Experimental: reduce excessive wrist roll around the forearm axis. Disabled by default because the "
            "correct twist axis depends on the target SMPL-X/FBX rig rest pose."
        ),
    )
    parser.add_argument(
        "--wrist_twist_limit_deg",
        type=float,
        default=150.0,
        help="Start softly damping wrist axial twist beyond this many degrees.",
    )
    parser.add_argument(
        "--wrist_twist_excess_keep",
        type=float,
        default=0.25,
        help="How much axial twist beyond the limit to keep. 0 clamps hard, 1 disables damping.",
    )
    return parser.parse_args()


def pick_device(device_arg):
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def to_np(data):
    if torch.is_tensor(data):
        return data.detach().cpu().numpy()
    return np.asarray(data)


def load_any_pkl(path):
    try:
        return torch.load(path, map_location="cpu")
    except Exception:
        with open(path, "rb") as f:
            return pickle.load(f)


def normalize_branch_dict(branch_dict):
    required = ["global_orient", "transl", "betas", "body_pose"]
    missing = [k for k in required if k not in branch_dict]
    if missing:
        raise KeyError(f"Missing keys in GVHMR branch: {missing}")

    normalized = {}
    for key in required:
        normalized[key] = to_np(branch_dict[key]).astype(np.float32).copy()

    num_frames = normalized["body_pose"].shape[0]
    if normalized["global_orient"].shape != (num_frames, 3):
        raise ValueError(f"Expected global_orient shape {(num_frames, 3)}, got {normalized['global_orient'].shape}")
    if normalized["transl"].shape != (num_frames, 3):
        raise ValueError(f"Expected transl shape {(num_frames, 3)}, got {normalized['transl'].shape}")
    if normalized["body_pose"].shape != (num_frames, 63):
        raise ValueError(f"Expected body_pose shape {(num_frames, 63)}, got {normalized['body_pose'].shape}")

    betas = normalized["betas"]
    if betas.ndim == 1:
        betas = np.repeat(betas[None], num_frames, axis=0)
    elif betas.shape[0] == 1 and num_frames > 1:
        betas = np.repeat(betas, num_frames, axis=0)
    normalized["betas"] = betas.astype(np.float32)
    return normalized


def resolve_gvhmr_branches(gvhmr_raw):
    branches = {}
    if isinstance(gvhmr_raw, dict) and "smpl_params_incam" in gvhmr_raw:
        branches["incam"] = normalize_branch_dict(gvhmr_raw["smpl_params_incam"])
    if isinstance(gvhmr_raw, dict) and "smpl_params_global" in gvhmr_raw:
        branches["global"] = normalize_branch_dict(gvhmr_raw["smpl_params_global"])

    if not branches:
        if not isinstance(gvhmr_raw, dict):
            raise TypeError("Unsupported GVHMR pkl structure")
        branches["flat"] = normalize_branch_dict(gvhmr_raw)

    return branches


def candidate_full_gvhmr_paths(args):
    candidates = []
    if args.gvhmr_full_pkl:
        candidates.append(Path(args.gvhmr_full_pkl))

    if not args.no_auto_full_gvhmr and args.video_path:
        video_stem = Path(args.video_path).stem
        candidates.extend(
            [
                REPO_ROOT / "outputs" / "demo" / video_stem / "hmr4d_results.pt",
                REPO_ROOT / "outputs" / "demo" / video_stem / "hmr4d_results.pkl",
            ]
        )

    if not args.no_auto_full_gvhmr:
        gvhmr_path = Path(args.gvhmr_pkl)
        candidates.extend(
            [
                gvhmr_path.with_name("hmr4d_results.pt"),
                gvhmr_path.parent / "hmr4d_results.pt",
            ]
        )

    seen = set()
    unique = []
    for path in candidates:
        path = path.expanduser()
        key = str(path.resolve()) if path.exists() else str(path)
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    return unique


def recover_full_gvhmr_if_available(args, branches):
    if "incam" in branches and "global" in branches:
        return branches

    for candidate in candidate_full_gvhmr_paths(args):
        if not candidate.exists():
            continue
        try:
            full_raw = load_any_pkl(candidate)
            full_branches = resolve_gvhmr_branches(full_raw)
        except Exception as exc:
            print(f"Warning: failed to inspect full GVHMR candidate {candidate}: {exc}")
            continue

        if "incam" in full_branches and "global" in full_branches:
            print(
                f"Using full GVHMR result with both camera/global branches: {candidate}\n"
                "This avoids mixing HaMeR camera-space hands with GVHMR DPVO/global body rotations."
            )
            return full_branches

    return branches


def choose_reference_branch(branches, request):
    if request != "auto":
        return choose_branch(branches, request, ["incam", "global", "flat"])

    if "incam" in branches:
        return "incam"

    if "flat" in branches:
        print(
            "Warning: only a flat GVHMR branch is available. HaMeR wrist global_orient is camera-space, "
            "but this flat pkl has no incam/global metadata. If this flat file came from smpl_params_global, "
            "wrist rotations can twist or pop by about 90 degrees. Prefer --gvhmr_full_pkl outputs/demo/<video>/hmr4d_results.pt."
        )
        return "flat"

    if "global" in branches:
        print(
            "Warning: using GVHMR global branch as wrist reference. HaMeR is camera-space, so this is only safe "
            "when the camera is static or the global branch is already camera-aligned."
        )
        return "global"

    return next(iter(branches))


def mat_to_rotvec_safe(mat):
    return R.from_matrix(mat).as_rotvec().astype(np.float32)


def rotvec_to_mat(rotvec):
    return R.from_rotvec(rotvec).as_matrix().astype(np.float32)


def project_to_rotation_matrix(rot):
    u, _, vh = np.linalg.svd(rot)
    rot_proj = u @ vh
    if np.linalg.det(rot_proj) < 0:
        u[:, -1] *= -1
        rot_proj = u @ vh
    return rot_proj.astype(np.float32)


def normalize_vec(vec, fallback):
    vec = np.asarray(vec, dtype=np.float32).reshape(3)
    norm = np.linalg.norm(vec)
    if norm < 1e-8:
        return np.asarray(fallback, dtype=np.float32)
    return (vec / norm).astype(np.float32)


def smplx_model_path_from_args(args):
    model_path = Path(args.smplx_model_dir)
    if model_path.is_file():
        return str(model_path)
    return str(model_path)


def estimate_wrist_twist_axes(args):
    fallback_axes = {
        LEFT_HAND_INDEX: np.array([1.0, 0.0, 0.0], dtype=np.float32),
        RIGHT_HAND_INDEX: np.array([-1.0, 0.0, 0.0], dtype=np.float32),
    }
    try:
        model = smplx.create(
            smplx_model_path_from_args(args),
            model_type="smplx",
            gender=args.gender,
            use_pca=False,
            flat_hand_mean=True,
            ext=args.model_ext,
        )
        model.eval()
        with torch.no_grad():
            output = model(
                betas=torch.zeros(1, 10),
                global_orient=torch.zeros(1, 3),
                body_pose=torch.zeros(1, 63),
                left_hand_pose=torch.zeros(1, 45),
                right_hand_pose=torch.zeros(1, 45),
                transl=torch.zeros(1, 3),
            )
        joints = output.joints[0, :22].detach().cpu().numpy()
        return {
            LEFT_HAND_INDEX: normalize_vec(joints[20] - joints[18], fallback_axes[LEFT_HAND_INDEX]),
            RIGHT_HAND_INDEX: normalize_vec(joints[21] - joints[19], fallback_axes[RIGHT_HAND_INDEX]),
        }
    except Exception as exc:
        print(f"Warning: failed to estimate SMPL-X forearm axes, using +/-X fallback: {exc}")
        return fallback_axes


def decompose_swing_twist(rot_mat, twist_axis):
    twist_axis = normalize_vec(twist_axis, [1.0, 0.0, 0.0])
    quat_xyzw = R.from_matrix(rot_mat).as_quat()
    vec = quat_xyzw[:3]
    scalar = quat_xyzw[3]
    projected = twist_axis * np.dot(vec, twist_axis)
    twist_norm = np.sqrt(float(np.dot(projected, projected) + scalar * scalar))
    if twist_norm < 1e-8:
        return rot_mat.astype(np.float32), 0.0

    twist_quat = np.concatenate([projected / twist_norm, [scalar / twist_norm]]).astype(np.float32)
    if twist_quat[3] < 0:
        twist_quat = -twist_quat

    twist_angle = 2.0 * np.arctan2(float(np.dot(twist_quat[:3], twist_axis)), float(twist_quat[3]))
    twist_mat = R.from_rotvec(twist_axis * twist_angle).as_matrix().astype(np.float32)
    swing_mat = project_to_rotation_matrix(rot_mat @ twist_mat.T)
    return swing_mat, twist_angle


def damp_wrist_twist_angle(angle, limit_rad, excess_keep):
    abs_angle = abs(float(angle))
    if abs_angle <= limit_rad:
        return float(angle)
    excess_keep = float(np.clip(excess_keep, 0.0, 1.0))
    damped = limit_rad + (abs_angle - limit_rad) * excess_keep
    return float(np.sign(angle) * damped)


def filter_wrist_twist_sequence(wrist_mats, hand_valid, twist_axes, limit_deg, excess_keep):
    filtered = wrist_mats.copy()
    limit_rad = np.deg2rad(limit_deg)
    for hand_idx in [LEFT_HAND_INDEX, RIGHT_HAND_INDEX]:
        valid_frames = np.flatnonzero(hand_valid[:, hand_idx])
        if len(valid_frames) == 0:
            continue

        axis = twist_axes[hand_idx]
        swings = []
        angles = []
        for frame_idx in valid_frames:
            swing, angle = decompose_swing_twist(wrist_mats[frame_idx, hand_idx], axis)
            swings.append(swing)
            angles.append(angle)

        angles = np.unwrap(np.asarray(angles, dtype=np.float32))
        angles = np.asarray([damp_wrist_twist_angle(a, limit_rad, excess_keep) for a in angles], dtype=np.float32)
        for local_idx, frame_idx in enumerate(valid_frames):
            twist_mat = R.from_rotvec(axis * angles[local_idx]).as_matrix().astype(np.float32)
            filtered[frame_idx, hand_idx] = project_to_rotation_matrix(swings[local_idx] @ twist_mat)

    return filtered


def compute_global_rotation(full_pose_axis_angle, joint_idx):
    global_rotation = np.eye(3, dtype=np.float32)
    while joint_idx != -1:
        joint_rotation = rotvec_to_mat(full_pose_axis_angle[joint_idx])
        global_rotation = joint_rotation @ global_rotation
        joint_idx = SMPLX_PARENTS_22[joint_idx]
    return global_rotation


def decode_hamer_export(hamer_raw):
    if isinstance(hamer_raw, dict) and "global_orient" in hamer_raw and "hand_pose" in hamer_raw:
        global_orient = to_np(hamer_raw["global_orient"])
        hand_pose = to_np(hamer_raw["hand_pose"])

        if global_orient.ndim != 4 or global_orient.shape[1:] != (2, 3, 3):
            raise ValueError(f"Unexpected HaMeR global_orient shape: {global_orient.shape}")
        if hand_pose.ndim != 5 or hand_pose.shape[1:] != (2, 15, 3, 3):
            raise ValueError(f"Unexpected HaMeR hand_pose shape: {hand_pose.shape}")

        valid = ~np.isclose(global_orient, 0.0).all(axis=(-1, -2))
        return global_orient.astype(np.float32), hand_pose.astype(np.float32), valid

    if isinstance(hamer_raw, list):
        num_frames = len(hamer_raw)
        global_orient = np.zeros((num_frames, 2, 3, 3), dtype=np.float32)
        hand_pose = np.zeros((num_frames, 2, 15, 3, 3), dtype=np.float32)
        valid = np.zeros((num_frames, 2), dtype=bool)

        for frame_idx, frame_data in enumerate(hamer_raw):
            for det in frame_data.get("detections", []):
                res = det.get("results")
                if not res:
                    continue

                hand_idx = RIGHT_HAND_INDEX if float(det["is_right"]) == 1.0 else LEFT_HAND_INDEX
                orient_flat = to_np(res["orient"]).reshape(-1)
                pose_flat = to_np(res["pose"]).reshape(-1)

                if orient_flat.size == 3:
                    orient_mat = rotvec_to_mat(orient_flat)
                elif orient_flat.size == 9:
                    orient_mat = orient_flat.reshape(3, 3).astype(np.float32)
                else:
                    raise ValueError(f"Unexpected detection orient size: {orient_flat.size}")

                if pose_flat.size == 45:
                    pose_mat = R.from_rotvec(pose_flat.reshape(15, 3)).as_matrix().astype(np.float32)
                elif pose_flat.size == 135:
                    pose_mat = pose_flat.reshape(15, 3, 3).astype(np.float32)
                else:
                    raise ValueError(f"Unexpected detection pose size: {pose_flat.size}")

                global_orient[frame_idx, hand_idx] = orient_mat
                hand_pose[frame_idx, hand_idx] = pose_mat
                valid[frame_idx, hand_idx] = True

        return global_orient, hand_pose, valid

    raise TypeError("Unsupported HaMeR pkl structure")


def choose_branch(branches, request, fallback_order):
    if request != "auto":
        if request not in branches:
            raise KeyError(f"Requested branch '{request}' not found. Available: {list(branches.keys())}")
        return request

    for name in fallback_order:
        if name in branches:
            return name
    return next(iter(branches))


def load_smplx_hand_means(smplx_model_dir, gender, ext):
    if os.path.isdir(smplx_model_dir):
        model_path = os.path.join(smplx_model_dir, f"SMPLX_{gender.upper()}.{ext}")
    else:
        model_path = smplx_model_dir

    if ext == "npz":
        data = np.load(model_path, allow_pickle=True)
    elif ext == "pkl":
        with open(model_path, "rb") as f:
            data = pickle.load(f, encoding="latin1")
    else:
        raise ValueError(f"Unsupported SMPL-X extension: {ext}")

    left_mean = np.asarray(data["hands_meanl"], dtype=np.float32).reshape(45)
    right_mean = np.asarray(data["hands_meanr"], dtype=np.float32).reshape(45)
    return left_mean, right_mean


def residual_mats_to_flat_abs_rotvec(residual_mats, hand_mean_rotvec):
    residual_rotvec = R.from_matrix(residual_mats).as_rotvec().astype(np.float32)
    abs_rotvec = residual_rotvec + hand_mean_rotvec.reshape(15, 3)
    abs_mats = R.from_rotvec(abs_rotvec).as_matrix().astype(np.float32)
    flat_rotvec = R.from_matrix(abs_mats).as_rotvec().astype(np.float32)
    return flat_rotvec.reshape(-1), abs_mats


def build_hand_pose_arrays(
    reference_branch,
    hamer_orient,
    hamer_hand_pose,
    hamer_valid,
    left_hand_mean=None,
    right_hand_mean=None,
    mirror_left_hand=True,
    bake_hand_mean=False,
    wrist_twist_mode="soft_clamp",
    wrist_twist_axes=None,
    wrist_twist_limit_deg=150.0,
    wrist_twist_excess_keep=0.25,
):
    num_frames = reference_branch["body_pose"].shape[0]
    hamer_frames = hamer_orient.shape[0]
    usable_frames = min(num_frames, hamer_frames)

    wrist_mats = np.tile(np.eye(3, dtype=np.float32), (num_frames, 2, 1, 1))
    wrist_body_pose = np.zeros((num_frames, 2, 3), dtype=np.float32)
    hand_pose_rotvec = np.zeros((num_frames, 2, 45), dtype=np.float32)
    hand_valid = np.zeros((num_frames, 2), dtype=bool)

    for frame_idx in tqdm(range(usable_frames), desc="Fusing body and hand rotations"):
        full_pose = np.concatenate(
            [reference_branch["global_orient"][frame_idx][None], reference_branch["body_pose"][frame_idx].reshape(21, 3)],
            axis=0,
        )
        left_elbow_global = compute_global_rotation(full_pose, 18)
        right_elbow_global = compute_global_rotation(full_pose, 19)

        if hamer_valid[frame_idx, LEFT_HAND_INDEX]:
            left_global = project_to_rotation_matrix(hamer_orient[frame_idx, LEFT_HAND_INDEX])
            left_residual_mats = np.stack(
                [project_to_rotation_matrix(m) for m in hamer_hand_pose[frame_idx, LEFT_HAND_INDEX]], axis=0
            )

            if bake_hand_mean:
                if right_hand_mean is None:
                    raise ValueError("--bake_hand_mean requires a SMPL-X model with hands_meanr")
                _, right_abs_mats = residual_mats_to_flat_abs_rotvec(left_residual_mats, right_hand_mean)
                left_fingers = LEFT_MIRROR[None] @ right_abs_mats @ LEFT_MIRROR
                left_rotvec = R.from_matrix(left_fingers).as_rotvec().reshape(-1).astype(np.float32)
            else:
                left_fingers = left_residual_mats
                left_rotvec = R.from_matrix(left_fingers).as_rotvec().reshape(-1).astype(np.float32)

            if mirror_left_hand:
                left_global = LEFT_MIRROR @ left_global @ LEFT_MIRROR
                if not bake_hand_mean:
                    left_fingers = LEFT_MIRROR[None] @ left_fingers @ LEFT_MIRROR
                    left_rotvec = R.from_matrix(left_fingers).as_rotvec().reshape(-1).astype(np.float32)

            left_local = np.linalg.inv(left_elbow_global) @ left_global
            wrist_mats[frame_idx, LEFT_HAND_INDEX] = project_to_rotation_matrix(left_local)
            hand_pose_rotvec[frame_idx, LEFT_HAND_INDEX] = left_rotvec
            hand_valid[frame_idx, LEFT_HAND_INDEX] = True

        if hamer_valid[frame_idx, RIGHT_HAND_INDEX]:
            right_global = project_to_rotation_matrix(hamer_orient[frame_idx, RIGHT_HAND_INDEX])
            right_local = np.linalg.inv(right_elbow_global) @ right_global
            wrist_mats[frame_idx, RIGHT_HAND_INDEX] = project_to_rotation_matrix(right_local)
            right_fingers = np.stack(
                [project_to_rotation_matrix(m) for m in hamer_hand_pose[frame_idx, RIGHT_HAND_INDEX]], axis=0
            )
            if bake_hand_mean:
                if right_hand_mean is None:
                    raise ValueError("--bake_hand_mean requires a SMPL-X model with hands_meanr")
                right_rotvec, _ = residual_mats_to_flat_abs_rotvec(right_fingers, right_hand_mean)
            else:
                right_rotvec = R.from_matrix(right_fingers).as_rotvec().reshape(-1).astype(np.float32)
            hand_pose_rotvec[frame_idx, RIGHT_HAND_INDEX] = right_rotvec
            hand_valid[frame_idx, RIGHT_HAND_INDEX] = True

    if wrist_twist_mode == "soft_clamp":
        if wrist_twist_axes is None:
            wrist_twist_axes = {
                LEFT_HAND_INDEX: np.array([1.0, 0.0, 0.0], dtype=np.float32),
                RIGHT_HAND_INDEX: np.array([-1.0, 0.0, 0.0], dtype=np.float32),
            }
        wrist_mats = filter_wrist_twist_sequence(
            wrist_mats,
            hand_valid,
            wrist_twist_axes,
            wrist_twist_limit_deg,
            wrist_twist_excess_keep,
        )

    for frame_idx in range(num_frames):
        for hand_idx in [LEFT_HAND_INDEX, RIGHT_HAND_INDEX]:
            if hand_valid[frame_idx, hand_idx]:
                wrist_body_pose[frame_idx, hand_idx] = mat_to_rotvec_safe(wrist_mats[frame_idx, hand_idx])

    return wrist_body_pose, hand_pose_rotvec, hand_valid


def apply_hands_to_branch(branch, wrist_body_pose, hand_pose_rotvec, hand_valid):
    merged = {k: v.copy() for k, v in branch.items()}
    num_frames = merged["body_pose"].shape[0]
    merged["left_hand_pose"] = np.zeros((num_frames, 45), dtype=np.float32)
    merged["right_hand_pose"] = np.zeros((num_frames, 45), dtype=np.float32)

    for frame_idx in range(num_frames):
        if hand_valid[frame_idx, LEFT_HAND_INDEX]:
            merged["body_pose"][frame_idx, LEFT_WRIST_BODY_SLICE] = wrist_body_pose[frame_idx, LEFT_HAND_INDEX]
            merged["left_hand_pose"][frame_idx] = hand_pose_rotvec[frame_idx, LEFT_HAND_INDEX]
        if hand_valid[frame_idx, RIGHT_HAND_INDEX]:
            merged["body_pose"][frame_idx, RIGHT_WRIST_BODY_SLICE] = wrist_body_pose[frame_idx, RIGHT_HAND_INDEX]
            merged["right_hand_pose"][frame_idx] = hand_pose_rotvec[frame_idx, RIGHT_HAND_INDEX]

    return merged


def apply_root_motion_mode_to_branch(branch, mode):
    if mode == "original":
        return branch

    processed = {k: v.copy() for k, v in branch.items()}
    transl = processed["transl"].copy()
    if transl.shape[0] == 0:
        return processed

    if mode == "inplace_xz":
        transl[:, 0] = transl[0, 0]
        transl[:, 2] = transl[0, 2]
    elif mode == "inplace_xyz":
        transl[:, :] = transl[0:1]
    else:
        raise ValueError(f"Unsupported root motion mode: {mode}")

    processed["transl"] = transl.astype(np.float32)
    return processed


def package_output(merged_branches, export_branch_name, hand_valid):
    export_data = {k: v.copy() for k, v in merged_branches[export_branch_name].items()}
    export_data["meta"] = {
        "export_branch": export_branch_name,
        "available_branches": list(merged_branches.keys()),
        "hand_valid": hand_valid.astype(np.uint8),
    }

    if "incam" in merged_branches:
        export_data["smpl_params_incam"] = {k: v.copy() for k, v in merged_branches["incam"].items()}
    if "global" in merged_branches:
        export_data["smpl_params_global"] = {k: v.copy() for k, v in merged_branches["global"].items()}

    return export_data


def add_coordinate_meta(export_data, reference_branch_name, hamer_frame="full_image_camera"):
    meta = export_data.setdefault("meta", {})
    meta["wrist_reference_branch"] = reference_branch_name
    meta["hamer_global_orient_frame"] = hamer_frame
    meta["coordinate_rule"] = (
        "HaMeR global_orient is interpreted in the original full-image camera frame. "
        "Wrist local rotations are solved against the GVHMR incam branch when available, "
        "then copied to all output branches because SMPL-X body_pose/hand_pose are local joint rotations."
    )


def add_wrist_twist_meta(export_data, args):
    meta = export_data.setdefault("meta", {})
    meta["root_motion_mode"] = args.root_motion_mode
    meta["wrist_twist_mode"] = args.wrist_twist_mode
    meta["wrist_twist_limit_deg"] = float(args.wrist_twist_limit_deg)
    meta["wrist_twist_excess_keep"] = float(args.wrist_twist_excess_keep)
    meta["wrist_twist_rule"] = (
        "When enabled, wrist local rotations are decomposed into palm swing and axial forearm twist. "
        "Only excessive axial twist is damped; finger articulation and most palm direction are preserved."
    )


def make_blender_pickle_compatible(obj):
    if isinstance(obj, dict):
        return {k: make_blender_pickle_compatible(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [make_blender_pickle_compatible(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(make_blender_pickle_compatible(v) for v in obj)
    if isinstance(obj, np.ndarray):
        if obj.dtype == np.bool_:
            return obj.astype(np.uint8).tolist()
        if np.issubdtype(obj.dtype, np.number):
            return np.asarray(obj, dtype=np.float32).tolist()
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def save_blender_compatible_pkl(data, out_pkl):
    out_path = Path(out_pkl)
    blender_path = out_path.with_name(f"{out_path.stem}_blender{out_path.suffix}")
    blender_data = make_blender_pickle_compatible(data)
    with open(blender_path, "wb") as f:
        pickle.dump(blender_data, f, protocol=4)
    return blender_path


def sanitize_fps(fps):
    try:
        fps = float(fps)
    except (TypeError, ValueError):
        return 30
    if not np.isfinite(fps) or fps <= 1e-6:
        return 30
    return int(np.clip(round(fps), 1, 240))


def resolve_smplx_model_path(model_path, gender, ext):
    model_path = Path(model_path)
    if model_path.is_file():
        return str(model_path)
    return str(model_path)


def render_overlay_video(args, merged_branch, device):
    try:
        from hmr4d.utils.geo.hmr_cam import estimate_K
        from hmr4d.utils.video_io_utils import get_video_lwh, get_video_reader, get_writer
        from hmr4d.utils.vis.renderer import Renderer
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Rendering requires the GVHMR render stack, including pytorch3d. "
            "Your current environment is missing a render dependency."
        ) from exc

    if not args.video_path:
        raise ValueError("--video_path is required unless --skip_render is set")

    video_path = Path(args.video_path)
    out_video_path = Path(args.out_pkl).with_suffix("").as_posix() + "_render.mp4"
    out_video_path = Path(out_video_path)

    smplx_model = smplx.create(
        resolve_smplx_model_path(args.smplx_model_dir, args.gender, args.model_ext),
        model_type="smplx",
        gender=args.gender,
        use_pca=False,
        flat_hand_mean=not args.bake_hand_mean,
        ext=args.model_ext,
    ).to(device)
    smplx_model.eval()

    faces = smplx_model.faces
    if isinstance(faces, np.ndarray):
        faces = torch.from_numpy(faces.astype(np.int64))

    total_frames, width, height = get_video_lwh(str(video_path))
    fps_cap = cv2.VideoCapture(str(video_path))
    fps = fps_cap.get(cv2.CAP_PROP_FPS)
    fps_cap.release()
    fps = sanitize_fps(fps)

    K = estimate_K(width, height).to(device)
    renderer = Renderer(width, height, device=str(device), faces=faces.to(device), K=K)
    reader = get_video_reader(str(video_path))
    writer = get_writer(str(out_video_path), fps=fps, crf=23)

    render_frames = min(total_frames, merged_branch["body_pose"].shape[0])
    if args.render_max_frames > 0:
        render_frames = min(render_frames, args.render_max_frames)

    z_min = float(np.nanmin(merged_branch["transl"][:render_frames, 2]))
    z_max = float(np.nanmax(merged_branch["transl"][:render_frames, 2]))
    if z_max <= 0 or z_min < 0:
        print(
            f"Warning: render branch transl z range is [{z_min:.3f}, {z_max:.3f}]. "
            "If the overlay is invisible, this GVHMR pkl is likely not in camera coordinates."
        )

    for frame_idx, img_rgb in tqdm(enumerate(reader), total=render_frames, desc="Rendering overlay"):
        if frame_idx >= render_frames:
            break

        with torch.no_grad():
            output = smplx_model(
                global_orient=torch.from_numpy(merged_branch["global_orient"][frame_idx : frame_idx + 1]).to(device),
                body_pose=torch.from_numpy(merged_branch["body_pose"][frame_idx : frame_idx + 1]).to(device),
                left_hand_pose=torch.from_numpy(merged_branch["left_hand_pose"][frame_idx : frame_idx + 1]).to(device),
                right_hand_pose=torch.from_numpy(merged_branch["right_hand_pose"][frame_idx : frame_idx + 1]).to(device),
                transl=torch.from_numpy(merged_branch["transl"][frame_idx : frame_idx + 1]).to(device),
                betas=torch.from_numpy(merged_branch["betas"][frame_idx : frame_idx + 1]).to(device),
            )

        verts = output.vertices[0]
        img_overlay = renderer.render_mesh(verts, img_rgb, [0.75, 0.78, 0.82])
        writer.write_frame(img_overlay)

    writer.close()
    if hasattr(reader, "close"):
        reader.close()
    return out_video_path


def main():
    args = parse_args()
    device = pick_device(args.device)

    print("Loading GVHMR and HaMeR pkls...")
    gvhmr_raw = load_any_pkl(args.gvhmr_pkl)
    hamer_raw = load_any_pkl(args.hamer_pkl)

    gvhmr_branches = resolve_gvhmr_branches(gvhmr_raw)
    gvhmr_branches = recover_full_gvhmr_if_available(args, gvhmr_branches)
    hamer_orient, hamer_hand_pose, hamer_valid = decode_hamer_export(hamer_raw)
    left_hand_mean, right_hand_mean = (None, None)
    if args.bake_hand_mean:
        left_hand_mean, right_hand_mean = load_smplx_hand_means(args.smplx_model_dir, args.gender, args.model_ext)

    reference_branch_name = choose_reference_branch(gvhmr_branches, args.wrist_reference_branch)
    reference_branch = gvhmr_branches[reference_branch_name]
    print(f"Wrist reference branch: {reference_branch_name}")
    wrist_twist_axes = estimate_wrist_twist_axes(args) if args.wrist_twist_mode != "none" else None
    if wrist_twist_axes is not None:
        print(
            "Wrist twist soft clamp: "
            f"limit={args.wrist_twist_limit_deg:.1f} deg, excess_keep={args.wrist_twist_excess_keep:.2f}"
        )

    wrist_body_pose, hand_pose_rotvec, hand_valid = build_hand_pose_arrays(
        reference_branch,
        hamer_orient,
        hamer_hand_pose,
        hamer_valid,
        left_hand_mean=left_hand_mean,
        right_hand_mean=right_hand_mean,
        mirror_left_hand=not args.disable_left_hand_mirror,
        bake_hand_mean=args.bake_hand_mean,
        wrist_twist_mode=args.wrist_twist_mode,
        wrist_twist_axes=wrist_twist_axes,
        wrist_twist_limit_deg=args.wrist_twist_limit_deg,
        wrist_twist_excess_keep=args.wrist_twist_excess_keep,
    )

    merged_branches = {}
    for branch_name, branch in gvhmr_branches.items():
        merged_branches[branch_name] = apply_hands_to_branch(branch, wrist_body_pose, hand_pose_rotvec, hand_valid)
        if args.root_motion_mode != "original" and branch_name in {"global", "flat"}:
            merged_branches[branch_name] = apply_root_motion_mode_to_branch(
                merged_branches[branch_name], args.root_motion_mode
            )

    export_branch_name = choose_branch(merged_branches, args.export_branch, ["global", "incam", "flat"])
    render_branch_name = choose_branch(merged_branches, args.render_branch, ["incam", "global", "flat"])

    export_data = package_output(merged_branches, export_branch_name, hand_valid)
    add_coordinate_meta(export_data, reference_branch_name)
    add_wrist_twist_meta(export_data, args)
    out_pkl = Path(args.out_pkl)
    out_pkl.parent.mkdir(parents=True, exist_ok=True)
    with open(out_pkl, "wb") as f:
        pickle.dump(export_data, f)
    blender_pkl = save_blender_compatible_pkl(export_data, out_pkl)

    left_count = int(hand_valid[:, LEFT_HAND_INDEX].sum())
    right_count = int(hand_valid[:, RIGHT_HAND_INDEX].sum())
    print(f"Saved merged pkl to: {out_pkl}")
    print(f"Saved Blender-compatible pkl to: {blender_pkl}")
    print(f"Valid hand frames - left: {left_count}, right: {right_count}")
    print(f"Top-level export branch: {export_branch_name}")
    print(f"Hand pose export mode: {'SMPL-X hand mean baked' if args.bake_hand_mean else 'mano2smplx direct matrices'}")

    if not args.skip_render:
        if device.type != "cuda":
            print("Warning: rendering on CPU may be slow.")
        out_video_path = render_overlay_video(args, merged_branches[render_branch_name], device)
        print(f"Saved overlay render to: {out_video_path}")
        print(f"Render branch: {render_branch_name}")


if __name__ == "__main__":
    main()
