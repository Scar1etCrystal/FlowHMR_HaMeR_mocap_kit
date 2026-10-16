"""
用 HandBody stitch_and_render.py 的合并逻辑替代 mano2smplx_merge.py，
输出与 Blender 插件兼容的 merged.pkl 格式。

用法:
  python hand_body_merge.py --gvhmr_pkl <hmr4d_results.pt> --hamer_pkl <hand_data.pkl> --output <merged.pkl>

依赖: D:\GVHMR-main\tools\demo\stitch_and_render.py 在 sys.path 上
"""

import argparse
import os
import pickle
import sys

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R, Slerp

# stitch_and_render.py is in the same directory (bundled)
_MERGE_DIR = os.path.dirname(os.path.abspath(__file__))
if _MERGE_DIR not in sys.path:
    sys.path.insert(0, _MERGE_DIR)

from stitch_and_render import (
    apply_hands_to_branch,
    build_hand_pose_arrays,
    decode_hamer_export,
    LEFT_HAND_INDEX,
    RIGHT_HAND_INDEX,
    resolve_gvhmr_branches,
)

# Import forearm twist transfer from our own merge module
from mano2smplx_merge import (
    forearm_twist_transfer, aa_to_rotmat, rotmat_to_aa,
    smooth_wrist_and_fingers, rotmat_to_quat, quat_to_rotmat,
)


def convert_hand_data_to_matrix_export(hand_data: dict):
    """将我们的 hand_data.pkl 转为 HandBody decode_hamer_export 兼容格式。"""
    n = hand_data["left_global_orient"].shape[0]
    global_orient = np.zeros((n, 2, 3, 3), dtype=np.float32)
    hand_pose = np.zeros((n, 2, 15, 3, 3), dtype=np.float32)

    for side, idx in [("left", 0), ("right", 1)]:
        detected = hand_data[f"{side}_is_detected"]
        orient_aa = hand_data[f"{side}_global_orient"]
        pose_aa = hand_data[f"{side}_hand_pose"]
        for i in np.where(detected)[0]:
            global_orient[i, idx] = R.from_rotvec(orient_aa[i]).as_matrix()
            hand_pose[i, idx] = R.from_rotvec(pose_aa[i].reshape(15, 3)).as_matrix()

    return {"global_orient": global_orient, "hand_pose": hand_pose}


def _slerp_rotmat_pair(a, b, fractions):
    """Interpolate one rotation matrix pair for the given fractions."""
    fractions = np.asarray(fractions, dtype=np.float64)
    if fractions.size == 0:
        return np.empty((0, 3, 3), dtype=np.float32)
    slerp = Slerp([0.0, 1.0], R.from_matrix(np.stack([a, b], axis=0)))
    return slerp(fractions).as_matrix().astype(np.float32)


def _copy_hand_frame(orient, hand_pose, valid, side, dst, src):
    orient[dst, side] = orient[src, side]
    hand_pose[dst, side] = hand_pose[src, side]
    valid[dst, side] = True


def _interp_hand_gap(orient, hand_pose, valid, side, start, end):
    frames = np.arange(start + 1, end, dtype=np.int64)
    if frames.size == 0:
        return 0
    fractions = (frames - start) / float(end - start)
    orient[frames, side] = _slerp_rotmat_pair(orient[start, side], orient[end, side], fractions)
    for joint_idx in range(hand_pose.shape[2]):
        hand_pose[frames, side, joint_idx] = _slerp_rotmat_pair(
            hand_pose[start, side, joint_idx],
            hand_pose[end, side, joint_idx],
            fractions,
        )
    valid[frames, side] = True
    return int(frames.size)


def fill_missing_hand_detections(orient, hand_pose, valid, hold_frames=6, interp_frames=15):
    """Fill short missing hand spans before wrist solving.

    Missing spans up to hold_frames copy the latest detected pose. Closed spans
    up to interp_frames are interpolated between their detected endpoints.
    Longer spans keep only a short hold after the last reliable detection.
    """
    hold_frames = max(0, int(hold_frames or 0))
    interp_frames = max(0, int(interp_frames or 0))
    if hold_frames == 0 and interp_frames == 0:
        return orient, hand_pose, valid, {"held": [0, 0], "interpolated": [0, 0]}

    orient = np.asarray(orient).copy()
    hand_pose = np.asarray(hand_pose).copy()
    valid = np.asarray(valid, dtype=bool).copy()
    n_frames = orient.shape[0]
    stats = {"held": [0, 0], "interpolated": [0, 0]}

    for side in (0, 1):
        detected = np.where(valid[:, side])[0]
        if detected.size == 0:
            continue

        for prev_idx, next_idx in zip(detected[:-1], detected[1:]):
            gap = int(next_idx - prev_idx - 1)
            if gap <= 0:
                continue
            if gap <= hold_frames:
                for frame_idx in range(prev_idx + 1, next_idx):
                    _copy_hand_frame(orient, hand_pose, valid, side, frame_idx, prev_idx)
                stats["held"][side] += gap
            elif interp_frames > 0 and gap <= interp_frames:
                stats["interpolated"][side] += _interp_hand_gap(
                    orient, hand_pose, valid, side, prev_idx, next_idx
                )
            else:
                hold_count = min(hold_frames, gap)
                for frame_idx in range(prev_idx + 1, prev_idx + 1 + hold_count):
                    _copy_hand_frame(orient, hand_pose, valid, side, frame_idx, prev_idx)
                stats["held"][side] += hold_count

        last_idx = int(detected[-1])
        trailing = n_frames - last_idx - 1
        hold_count = min(hold_frames, trailing)
        for frame_idx in range(last_idx + 1, last_idx + 1 + hold_count):
            _copy_hand_frame(orient, hand_pose, valid, side, frame_idx, last_idx)
        stats["held"][side] += int(hold_count)

    return orient, hand_pose, valid, stats


def _rot_angle_deg(a, b):
    rel = np.matmul(np.asarray(a).T, np.asarray(b))
    return float(np.degrees(R.from_matrix(rel).magnitude()))


def _pose_delta_deg(a, b):
    angles = [_rot_angle_deg(a[i], b[i]) for i in range(a.shape[0])]
    return float(np.mean(angles)), float(np.max(angles))


def filter_hand_jump_frames(orient, hand_pose, valid, wrist_max_deg=160.0,
                            finger_mean_max_deg=90.0, finger_max_deg=170.0):
    """Mark implausible hand pose jumps as missing before gap filling."""
    valid = np.asarray(valid, dtype=bool).copy()
    stats = {"rejected": [0, 0], "kept": [0, 0]}
    for side in (0, 1):
        last_orient = None
        last_pose = None
        for frame_idx in range(valid.shape[0]):
            if not valid[frame_idx, side]:
                continue
            accept = True
            if last_orient is not None:
                wrist_delta = _rot_angle_deg(last_orient, orient[frame_idx, side])
                finger_mean, finger_max = _pose_delta_deg(last_pose, hand_pose[frame_idx, side])
                accept = (
                    wrist_delta <= wrist_max_deg
                    and finger_mean <= finger_mean_max_deg
                    and finger_max <= finger_max_deg
                )
            if accept:
                last_orient = orient[frame_idx, side]
                last_pose = hand_pose[frame_idx, side]
                stats["kept"][side] += 1
            else:
                valid[frame_idx, side] = False
                stats["rejected"][side] += 1
    return valid, stats


def main():
    parser = argparse.ArgumentParser(description="Hand-body merge → Blender-compatible merged.pkl")
    parser.add_argument("--gvhmr_pkl", required=True, help="hmr4d_results.pt or .pkl")
    parser.add_argument("--hamer_pkl", required=True, help="hand_data.pkl (our format)")
    parser.add_argument("--output", required=True, help="Output merged.pkl path")
    parser.add_argument("--mirror_left_hand", action="store_true", default=True, help="Mirror left hand (default: True)")
    parser.add_argument("--no_mirror_left_hand", action="store_true", help="Don't mirror left hand")
    parser.add_argument("--bake_hand_mean", action="store_true", help="Bake SMPL-X hand mean pose")
    parser.add_argument("--wrist_twist_mode", default="none", choices=["none", "soft"], help="Wrist twist mode")
    # Smoothing / limits (same as mano2smplx_merge)
    parser.add_argument("--smooth", type=str, default="none",
                        choices=["one_euro", "butterworth", "both", "none"],
                        help="Smoothing method (default: none, raw hand-body merge)")
    parser.add_argument("--wrist_cutoff", type=float, default=1.0,
                        help="One-Euro wrist min cutoff frequency (lower = smoother)")
    parser.add_argument("--wrist_beta", type=float, default=0.007,
                        help="One-Euro wrist speed coefficient")
    parser.add_argument("--finger_cutoff", type=float, default=2.0,
                        help="One-Euro finger min cutoff frequency")
    parser.add_argument("--finger_beta", type=float, default=0.01,
                        help="One-Euro finger speed coefficient")
    parser.add_argument("--no_finger_smooth", action="store_true",
                        help="Disable finger smoothing")
    parser.add_argument("--max_jump_deg", type=float, default=90.0,
                        help="Max quaternion jump in degrees for unwrap (0=disabled)")
    parser.add_argument("--forearm_twist", action="store_true", default=False,
                        help="Enable forearm twist transfer")
    parser.add_argument("--gvhmr_fps", type=float, default=None,
                        help="Source video fps for time-aligned smoothing")
    parser.add_argument("--hold_missing_frames", type=int, default=6,
                        help="Copy the latest detected hand pose across gaps up to this many frames")
    parser.add_argument("--interp_missing_frames", type=int, default=15,
                        help="Interpolate closed missing hand gaps up to this many frames (0 disables)")
    parser.add_argument("--no_fill_missing_hands", action="store_true",
                        help="Disable missing hand gap filling before merge")
    parser.add_argument("--filter_jump_frames", action="store_true",
                        help="Mark implausible hand pose jumps as missing before gap filling")
    parser.add_argument("--jump_wrist_max_deg", type=float, default=160.0,
                        help="Reject a hand frame when wrist delta from last accepted frame exceeds this")
    parser.add_argument("--jump_finger_mean_max_deg", type=float, default=90.0,
                        help="Reject a hand frame when mean finger-joint delta exceeds this")
    parser.add_argument("--jump_finger_max_deg", type=float, default=170.0,
                        help="Reject a hand frame when any finger-joint delta exceeds this")
    args = parser.parse_args()

    mirror_left = args.mirror_left_hand and not args.no_mirror_left_hand

    # ---- Load ----
    print(f"[HandBodyMerge] Loading GVHMR: {args.gvhmr_pkl}")
    gvhmr_pkl_path = str(args.gvhmr_pkl)
    if gvhmr_pkl_path.endswith((".pt", ".pth")):
        gvhmr_raw = torch.load(gvhmr_pkl_path, map_location="cpu", weights_only=False)
    else:
        with open(gvhmr_pkl_path, "rb") as f:
            gvhmr_raw = pickle.load(f)

    print(f"[HandBodyMerge] Loading HAMER: {args.hamer_pkl}")
    with open(args.hamer_pkl, "rb") as f:
        hamer_raw = pickle.load(f)

    # Detect HAMER format and convert to HandBody's (N,2,3,3) matrix format
    if isinstance(hamer_raw, dict) and "left_global_orient" in hamer_raw:
        # Image mode: our dict format with axis-angle per side
        matrix_hamer = convert_hand_data_to_matrix_export(hamer_raw)
        hamer_orient, hamer_hand_pose, hamer_valid = decode_hamer_export(matrix_hamer)
    elif isinstance(hamer_raw, dict) and "global_orient" in hamer_raw and "hand_pose" in hamer_raw:
        # Video mode (hamer_video.py): already in matrix-export (N,2,...) matrix format
        hamer_orient, hamer_hand_pose, hamer_valid = decode_hamer_export(hamer_raw)
    elif isinstance(hamer_raw, list):
        # Legacy list format from original HAMER
        hamer_orient, hamer_hand_pose, hamer_valid = decode_hamer_export(hamer_raw)
    else:
        raise ValueError(f"Unrecognized HAMER pkl format. Keys: {list(hamer_raw.keys()) if isinstance(hamer_raw, dict) else type(hamer_raw)}")

    n_frames_hamer = hamer_orient.shape[0]
    print(f"[HandBodyMerge] HAMER frames: {n_frames_hamer}, "
          f"L valid: {hamer_valid[:, 0].sum()}, R valid: {hamer_valid[:, 1].sum()}")

    if args.filter_jump_frames:
        hamer_valid, jump_stats = filter_hand_jump_frames(
            hamer_orient,
            hamer_hand_pose,
            hamer_valid,
            wrist_max_deg=args.jump_wrist_max_deg,
            finger_mean_max_deg=args.jump_finger_mean_max_deg,
            finger_max_deg=args.jump_finger_max_deg,
        )
        print("[HandBodyMerge] Hand jump filter: "
              f"rejected L/R={jump_stats['rejected'][0]}/{jump_stats['rejected'][1]}, "
              f"valid now L/R={hamer_valid[:, 0].sum()}/{hamer_valid[:, 1].sum()}")

    if not args.no_fill_missing_hands:
        hamer_orient, hamer_hand_pose, hamer_valid, fill_stats = fill_missing_hand_detections(
            hamer_orient,
            hamer_hand_pose,
            hamer_valid,
            hold_frames=args.hold_missing_frames,
            interp_frames=args.interp_missing_frames,
        )
        print("[HandBodyMerge] Missing hand fill: "
              f"hold L/R={fill_stats['held'][0]}/{fill_stats['held'][1]}, "
              f"interp L/R={fill_stats['interpolated'][0]}/{fill_stats['interpolated'][1]}, "
              f"valid now L/R={hamer_valid[:, 0].sum()}/{hamer_valid[:, 1].sum()}")

    # ---- Resolve GVHMR branches ----
    gvhmr_branches = resolve_gvhmr_branches(gvhmr_raw)
    print(f"[HandBodyMerge] GVHMR branches: {list(gvhmr_branches.keys())}")
    for name, b in gvhmr_branches.items():
        print(f"  {name}: body_pose {b['body_pose'].shape}")

    # Pick reference branch (prefer incam for wrist solving, like the default merge path)
    ref_name = "incam" if "incam" in gvhmr_branches else list(gvhmr_branches.keys())[0]
    ref_branch = gvhmr_branches[ref_name]
    print(f"[HandBodyMerge] Reference branch: {ref_name}")

    # ---- Compensate HAMER orient Z-axis 180° offset ----
    # The hand-body wrist solution has a systematic flip vs Blender SMPL-X convention.
    # Original output needed Y+180°; after applying Ry180, X was still off by 180°.
    # Combined correction: Rx180 @ Ry180 = Rz180 = diag(-1, -1, 1).
    RZ180 = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]], dtype=np.float32)
    hamer_orient = np.einsum("ij,...fjk->...fik", RZ180, hamer_orient)

    # ---- Run hand-body merge ----
    wrist_body_pose, hand_pose_rotvec, hand_valid = build_hand_pose_arrays(
        ref_branch,
        hamer_orient,
        hamer_hand_pose,
        hamer_valid,
        left_hand_mean=None,
        right_hand_mean=None,
        mirror_left_hand=mirror_left,
        bake_hand_mean=args.bake_hand_mean,
        wrist_twist_mode=args.wrist_twist_mode,
    )

    # ---- Apply to all branches ----
    merged_branches = {}
    for branch_name, branch in gvhmr_branches.items():
        merged = apply_hands_to_branch(branch, wrist_body_pose, hand_pose_rotvec, hand_valid)
        merged_branches[branch_name] = merged

    # ---- Post-processing: smoothing + anatomical limits ----
    smooth_method = args.smooth
    do_smooth = smooth_method != "none" or args.max_jump_deg > 0 or args.forearm_twist
    fps = args.gvhmr_fps if args.gvhmr_fps else 30.0

    if do_smooth:
        print(f"[HandBodyMerge] Post-processing: smooth={smooth_method}, fps={fps}, "
              f"max_jump={args.max_jump_deg}°, forearm_twist={args.forearm_twist}")

        for branch_name, branch in merged_branches.items():
            bp = branch["body_pose"]  # (N, 63) — axis-angle
            lhp = branch["left_hand_pose"]   # (N, 45) — axis-angle
            rhp = branch["right_hand_pose"]  # (N, 45) — axis-angle
            go = branch["global_orient"]     # (N, 3) — axis-angle
            N = bp.shape[0]

            # Build wrist rotmats from body_pose wrist joints (indices 19,20 = slices [57:60],[60:63])
            left_wrist_rms = []
            right_wrist_rms = []
            for f in range(N):
                if hand_valid[f, 0] if f < hand_valid.shape[0] else False:
                    left_wrist_rms.append(aa_to_rotmat(bp[f, 57:60]))
                else:
                    left_wrist_rms.append(None)
                if hand_valid[f, 1] if f < hand_valid.shape[0] else False:
                    right_wrist_rms.append(aa_to_rotmat(bp[f, 60:63]))
                else:
                    right_wrist_rms.append(None)

            l_det = np.zeros(N, dtype=bool)
            r_det = np.zeros(N, dtype=bool)
            for f in range(min(N, hand_valid.shape[0])):
                l_det[f] = bool(hand_valid[f, 0])
                r_det[f] = bool(hand_valid[f, 1])

            bp_new, lhp_new, rhp_new = smooth_wrist_and_fingers(
                bp.copy(), lhp.copy(), rhp.copy(),
                left_wrist_rms, right_wrist_rms,
                l_det, r_det,
                fps=fps,
                smooth_method=smooth_method,
                one_euro_wrist_min_cutoff=args.wrist_cutoff,
                one_euro_wrist_beta=args.wrist_beta,
                one_euro_finger_min_cutoff=args.finger_cutoff,
                one_euro_finger_beta=args.finger_beta,
                max_jump_deg=args.max_jump_deg,
                smooth_fingers=not args.no_finger_smooth,
                global_orient_arr=go if args.forearm_twist else None,
                forearm_twist=args.forearm_twist,
            )

            # Write smoothed data back
            branch["body_pose"] = bp_new
            branch["left_hand_pose"] = lhp_new
            branch["right_hand_pose"] = rhp_new

        print("[HandBodyMerge] Post-processing complete")
    else:
        print("[HandBodyMerge] No post-processing (raw merge)")

    # ---- Build Blender-compatible output ----
    # Our Blender addon expects:
    #   smpl_params_global: {body_pose, global_orient, transl, betas, left/right_hand_pose,
    #                        left/right_hand_global_orient, left/right_hand_is_detected}
    #   smpl_params_incam:  {same}
    #   hamer_merge_info: {source, method, ...}

    # Get the HAMER global_orient back in axis-angle for Blender addon's apply_hand_pose
    # (Blender addon uses it when use_hamer_wrist=True, which is currently forced False,
    #  but we include it for completeness)
    hamer_orient_aa = np.zeros((hand_valid.shape[0], 2, 3), dtype=np.float32)
    for side_idx in [0, 1]:
        for i in range(hand_valid.shape[0]):
            if hand_valid[i, side_idx]:
                hamer_orient_aa[i, side_idx] = R.from_matrix(hamer_orient[i, side_idx]).as_rotvec()

    output = {}

    for branch_name in ["global", "incam"]:
        if branch_name not in merged_branches:
            continue
        branch = merged_branches[branch_name]
        n = branch["body_pose"].shape[0]
        n_hamer = hand_valid.shape[0]

        params = {
            "body_pose": branch["body_pose"],
            "global_orient": branch["global_orient"],
            "transl": branch["transl"],
            "betas": branch["betas"],
            "left_hand_pose": branch["left_hand_pose"],
            "right_hand_pose": branch["right_hand_pose"],
        }

        # Pad HAMER-specific fields to match GVHMR frame count
        if n_hamer < n:
            pad = n - n_hamer
            l_go = np.pad(hamer_orient_aa[:n_hamer, 0], ((0, pad), (0, 0)))
            r_go = np.pad(hamer_orient_aa[:n_hamer, 1], ((0, pad), (0, 0)))
            l_det = np.pad(hand_valid[:n_hamer, 0], (0, pad))
            r_det = np.pad(hand_valid[:n_hamer, 1], (0, pad))
        else:
            l_go = hamer_orient_aa[:n, 0]
            r_go = hamer_orient_aa[:n, 1]
            l_det = hand_valid[:n, 0]
            r_det = hand_valid[:n, 1]

        params["left_hand_global_orient"] = l_go
        params["right_hand_global_orient"] = r_go
        params["left_hand_is_detected"] = l_det
        params["right_hand_is_detected"] = r_det

        output[f"smpl_params_{branch_name}"] = params

    output["hamer_merge_info"] = {
        "source": "stitch_and_render",
        "method": "build_hand_pose_arrays",
        "wrist_mode": "stitch",
        "reference_branch": ref_name,
        "mirror_left_hand": mirror_left,
        "gvhmr_frames": gvhmr_branches[ref_name]["body_pose"].shape[0],
        "hamer_frames_original": n_frames_hamer,
        "merged_frames": merged_branches[ref_name]["body_pose"].shape[0],
    }

    # ---- Save ----
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "wb") as f:
        pickle.dump(output, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"[HandBodyMerge] Saved: {args.output}")
    print(f"  Branches: {list(output.keys())}")
    n = merged_branches[ref_name]["body_pose"].shape[0]
    print(f"  Frames: {n}")
    print(f"  Left hand valid: {hand_valid[:, 0].sum()}")
    print(f"  Right hand valid: {hand_valid[:, 1].sum()}")


if __name__ == "__main__":
    main()
