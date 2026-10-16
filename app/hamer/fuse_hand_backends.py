import argparse
import os
import pickle

import numpy as np
from scipy.spatial.transform import Rotation as R


def _axis_angle_export_to_mats(hand_data):
    n = hand_data["left_global_orient"].shape[0]
    orient = np.zeros((n, 2, 3, 3), dtype=np.float32)
    pose = np.zeros((n, 2, 15, 3, 3), dtype=np.float32)
    valid = np.zeros((n, 2), dtype=bool)
    for side, side_idx in (("left", 0), ("right", 1)):
        detected = np.asarray(hand_data.get(f"{side}_is_detected", np.zeros(n)), dtype=bool)
        go = np.asarray(hand_data[f"{side}_global_orient"])
        hp = np.asarray(hand_data[f"{side}_hand_pose"])
        for frame_idx in np.where(detected)[0]:
            orient[frame_idx, side_idx] = R.from_rotvec(go[frame_idx]).as_matrix()
            pose[frame_idx, side_idx] = R.from_rotvec(hp[frame_idx].reshape(15, 3)).as_matrix()
            valid[frame_idx, side_idx] = True
    return orient, pose, valid


def _load_hand_pkl(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    if isinstance(data, dict) and "global_orient" in data and "hand_pose" in data:
        orient = np.asarray(data["global_orient"], dtype=np.float32)
        pose = np.asarray(data["hand_pose"], dtype=np.float32)
        valid = np.asarray(data.get("global_orient_valid", np.any(orient != 0, axis=(-1, -2))), dtype=bool)
        return data, orient, pose, valid
    if isinstance(data, dict) and "left_global_orient" in data:
        orient, pose, valid = _axis_angle_export_to_mats(data)
        return data, orient, pose, valid
    raise ValueError(f"Unsupported hand pkl format: {path}")


def _rot_angle_deg(a, b):
    rel = np.matmul(np.asarray(a).T, np.asarray(b))
    return float(np.degrees(R.from_matrix(rel).magnitude()))


def _pose_delta_deg(a, b):
    angles = [_rot_angle_deg(a[i], b[i]) for i in range(a.shape[0])]
    return float(np.mean(angles)), float(np.max(angles))


def _copy_side(dst, orient, pose, valid, frame_idx, side_idx, source):
    dst["global_orient"][frame_idx, side_idx] = orient[frame_idx, side_idx]
    dst["hand_pose"][frame_idx, side_idx] = pose[frame_idx, side_idx]
    dst["global_orient_valid"][frame_idx, side_idx] = 1
    dst["source_mask"][frame_idx, side_idx] = source


def fuse_hands(wilor_pkl, hamer_pkl, output, wrist_max_deg=80.0, finger_mean_max_deg=55.0, finger_max_deg=130.0):
    wilor_raw, wilor_orient, wilor_pose, wilor_valid = _load_hand_pkl(wilor_pkl)
    _, hamer_orient, hamer_pose, hamer_valid = _load_hand_pkl(hamer_pkl)

    n = max(wilor_orient.shape[0], hamer_orient.shape[0])
    result = {
        "global_orient": np.zeros((n, 2, 3, 3), dtype=np.float32),
        "hand_pose": np.zeros((n, 2, 15, 3, 3), dtype=np.float32),
        "global_orient_valid": np.zeros((n, 2), dtype=np.uint8),
        "source_mask": np.zeros((n, 2), dtype=np.uint8),  # 0=missing, 1=WiLoR, 2=HAMER fallback
        "source": "WiLoR+HAMER fusion",
        "fusion_info": {
            "wrist_max_deg": float(wrist_max_deg),
            "finger_mean_max_deg": float(finger_mean_max_deg),
            "finger_max_deg": float(finger_max_deg),
            "source_mask": "0=missing, 1=WiLoR, 2=HAMER fallback",
        },
    }
    for key in ("fps", "width", "height"):
        if isinstance(wilor_raw, dict) and key in wilor_raw:
            result[key] = wilor_raw[key]
    if isinstance(wilor_raw, dict) and "frames" in wilor_raw:
        result["frames"] = wilor_raw["frames"]

    stats = {
        "wilor": [0, 0],
        "hamer_used": [0, 0],
        "hamer_rejected": [0, 0],
        "missing": [0, 0],
    }
    last_orient = [None, None]
    last_pose = [None, None]

    for frame_idx in range(n):
        for side_idx in (0, 1):
            if frame_idx < wilor_valid.shape[0] and wilor_valid[frame_idx, side_idx]:
                _copy_side(result, wilor_orient, wilor_pose, wilor_valid, frame_idx, side_idx, 1)
                last_orient[side_idx] = wilor_orient[frame_idx, side_idx]
                last_pose[side_idx] = wilor_pose[frame_idx, side_idx]
                stats["wilor"][side_idx] += 1
                continue

            h_valid = frame_idx < hamer_valid.shape[0] and hamer_valid[frame_idx, side_idx]
            if not h_valid:
                stats["missing"][side_idx] += 1
                continue

            accept = False
            if last_orient[side_idx] is None or n <= 1:
                accept = True
            else:
                wrist_delta = _rot_angle_deg(last_orient[side_idx], hamer_orient[frame_idx, side_idx])
                finger_mean, finger_max = _pose_delta_deg(last_pose[side_idx], hamer_pose[frame_idx, side_idx])
                accept = (
                    wrist_delta <= wrist_max_deg
                    and finger_mean <= finger_mean_max_deg
                    and finger_max <= finger_max_deg
                )

            if accept:
                _copy_side(result, hamer_orient, hamer_pose, hamer_valid, frame_idx, side_idx, 2)
                last_orient[side_idx] = hamer_orient[frame_idx, side_idx]
                last_pose[side_idx] = hamer_pose[frame_idx, side_idx]
                stats["hamer_used"][side_idx] += 1
            else:
                stats["hamer_rejected"][side_idx] += 1

    result["fusion_info"]["stats"] = stats
    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    with open(output, "wb") as f:
        pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[Hand Fusion] Saved: {output}")
    print(
        "[Hand Fusion] "
        f"WiLoR L/R={stats['wilor'][0]}/{stats['wilor'][1]}, "
        f"HAMER used L/R={stats['hamer_used'][0]}/{stats['hamer_used'][1]}, "
        f"HAMER rejected L/R={stats['hamer_rejected'][0]}/{stats['hamer_rejected'][1]}, "
        f"missing L/R={stats['missing'][0]}/{stats['missing'][1]}"
    )


def main():
    parser = argparse.ArgumentParser(description="Fuse WiLoR primary hands with filtered HAMER fallback.")
    parser.add_argument("--wilor_pkl", required=True)
    parser.add_argument("--hamer_pkl", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--wrist_max_deg", type=float, default=80.0)
    parser.add_argument("--finger_mean_max_deg", type=float, default=55.0)
    parser.add_argument("--finger_max_deg", type=float, default=130.0)
    args = parser.parse_args()
    fuse_hands(
        args.wilor_pkl,
        args.hamer_pkl,
        args.output,
        wrist_max_deg=args.wrist_max_deg,
        finger_mean_max_deg=args.finger_mean_max_deg,
        finger_max_deg=args.finger_max_deg,
    )


if __name__ == "__main__":
    main()
