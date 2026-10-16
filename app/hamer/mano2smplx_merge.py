"""
Merge HAMER hand data into GVHMR body data using the mano2smplx approach.

This replaces merge_hand_body_wrist_solver_v6.py with a simpler, more direct
method based on the HAMER official mano2smplx.py:

1. Compute elbow global rotation from GVHMR body kinematic tree
2. HAMER global_orient = wrist rotation in hand-crop camera space (global)
3. Wrist local rotation = inv(elbow_global) @ wrist_global
4. Left hand mirror: M @ R @ M where M = diag(-1, 1, 1)
5. Finger poses: rotation matrices -> axis-angle (left mirror)

The merged output has the same format as v6 for compatibility with the
Blender import script.

Usage:
    python mano2smplx_merge.py --gvhmr_pkl body.pkl --hamer_pkl hand_data.pkl --output merged.pkl
"""

import argparse
import pickle
import numpy as np
import os
import copy


# SMPL 22-joint kinematic tree (same as mano2smplx.py)
SMPL_PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19]

# Left hand mirror matrix
M_MIRROR = np.diag([-1.0, 1.0, 1.0])

# Y-axis 180° rotation: corrects HAMER hand-crop camera space to body space
RY180 = np.array([[-1.0, 0, 0], [0, 1.0, 0], [0, 0, -1.0]], dtype=np.float64)

# Z-axis 180° bind correction: MANO hand rest pose -> SMPL wrist rest pose
RZ180 = np.array([[-1.0, 0, 0], [0, -1.0, 0], [0, 0, 1.0]], dtype=np.float64)


def rotmat_to_quat(R):
    """Rotation matrix (3,3) -> quaternion (w, x, y, z) using Shepperd's method."""
    tr = np.trace(R)
    if tr > 0:
        s = 2.0 * np.sqrt(tr + 1.0)
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z], dtype=np.float64)
    q /= np.linalg.norm(q)
    return q


def quat_to_rotmat(q):
    """Quaternion (w, x, y, z) -> rotation matrix (3,3)."""
    w, x, y, z = q
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w),   2*(x*z+y*w)],
        [2*(x*y+z*w),   1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w),   2*(y*z+x*w),   1-2*(x*x+y*y)],
    ], dtype=np.float64)


def geodesic_dist(R1, R2):
    """Geodesic distance between two rotation matrices in radians."""
    # (tr(R1^T @ R2) - 1) / 2  clamped to [-1, 1]
    val = np.clip((np.trace(R1.T @ R2) - 1.0) / 2.0, -1.0, 1.0)
    return np.arccos(val)


def quat_flip_to_minimize(q, q_prev):
    """Flip q to -q if that reduces geodesic distance to q_prev."""
    # For unit quaternions, distance = arccos(|q1·q2|)
    dot = abs(np.dot(q, q_prev))
    dot_neg = abs(np.dot(-q, q_prev))
    if dot_neg > dot:
        return -q
    return q


def clamp_wrist_per_axis(wrist_rotmats, x_limit_deg=45, y_limit_deg=45, z_limit_deg=90):
    """
    Clamp wrist rotation per-axis (X, Y, Z) in Euler angle space.
    Applied BEFORE unwrap to prevent extreme out-of-range rotations.
    
    Args:
        wrist_rotmats: list of (3,3) rotation matrices or None, one per frame
        x_limit_deg: max absolute rotation around X axis (degrees)
        y_limit_deg: max absolute rotation around Y axis (degrees)
        z_limit_deg: max absolute rotation around Z axis (degrees)
    
    Returns:
        new list of (3,3) rotation matrices with per-axis clamping applied
    """
    from scipy.spatial.transform import Rotation
    
    x_lim = np.radians(x_limit_deg)
    y_lim = np.radians(y_limit_deg)
    z_lim = np.radians(z_limit_deg)
    
    result = []
    for R in wrist_rotmats:
        if R is None:
            result.append(None)
            continue
        euler = Rotation.from_matrix(R).as_euler('XYZ')  # XYZ intrinsic
        clamped = np.array([
            np.clip(euler[0], -x_lim, x_lim),
            np.clip(euler[1], -y_lim, y_lim),
            np.clip(euler[2], -z_lim, z_lim),
        ])
        if not np.allclose(euler, clamped, atol=1e-8):
            R_clamped = Rotation.from_euler('XYZ', clamped).as_matrix()
            result.append(R_clamped)
        else:
            result.append(R)
    return result


def unwrap_wrist_rotations(wrist_rotmats, max_jump_deg=90):
    """
    Remove 180° flips and limit frame-to-frame jumps in wrist rotations.
    
    Two-stage approach:
    1. Quaternion sign consistency: flip q -> -q if it reduces geodesic distance
    2. Hard limit: if geodesic distance > max_jump_deg from previous (valid) frame,
       blend towards previous frame to cap the jump
    
    Args:
        wrist_rotmats: list of (3,3) rotation matrices, one per frame
        max_jump_deg: maximum allowed per-frame rotation change in degrees
    
    Returns:
        list of (3,3) rotation matrices with consistent representation
    """
    max_jump_rad = np.radians(max_jump_deg)
    N = len(wrist_rotmats)
    if N <= 1:
        return wrist_rotmats
    
    # Find first valid (non-None) frame as anchor
    first_valid = None
    for idx in range(N):
        if wrist_rotmats[idx] is not None:
            first_valid = idx
            break
    if first_valid is None:
        # All frames are None
        return wrist_rotmats
    
    result = [None] * N
    result[first_valid] = wrist_rotmats[first_valid].copy()
    prev_q = rotmat_to_quat(wrist_rotmats[first_valid])
    
    # Forward pass: from first_valid+1 to end
    for i in range(first_valid + 1, N):
        R = wrist_rotmats[i]
        if R is None:
            result[i] = result[i - 1].copy()
            prev_q = rotmat_to_quat(result[i])
            continue
        
        q = rotmat_to_quat(R)
        
        # Stage 1: flip sign if it brings us closer
        q = quat_flip_to_minimize(q, prev_q)
        
        # Stage 2: check geodesic distance, clamp if too large
        R_candidate = quat_to_rotmat(q)
        dist = geodesic_dist(quat_to_rotmat(prev_q), R_candidate)
        
        if dist > max_jump_rad:
            # SLERP blend: go max_jump_rad towards the new rotation
            # t = max_jump_rad / dist means we travel exactly max_jump_rad
            t = max_jump_rad / dist
            # SLERP: q_result = slerp(prev_q, q, t)
            dot = np.dot(prev_q, q)
            if dot < 0:
                q = -q
                dot = -dot
            dot = np.clip(dot, -1.0, 1.0)
            theta = np.arccos(dot)
            if theta < 1e-6:
                q_blended = prev_q.copy()
            else:
                q_blended = (np.sin((1-t)*theta) * prev_q + np.sin(t*theta) * q) / np.sin(theta)
            q_blended /= np.linalg.norm(q_blended)
            R_candidate = quat_to_rotmat(q_blended)
            q = q_blended
        
        result[i] = R_candidate
        prev_q = q.copy()
    
    # Backward pass: fill frames before first_valid with nearest valid value
    if first_valid > 0:
        fill_R = result[first_valid].copy()
        for i in range(first_valid - 1, -1, -1):
            if wrist_rotmats[i] is not None:
                R = wrist_rotmats[i]
                q = rotmat_to_quat(R)
                q = quat_flip_to_minimize(q, prev_q)
                R_candidate = quat_to_rotmat(q)
                result[i] = R_candidate
                prev_q = q.copy()
            else:
                result[i] = fill_R
    
    return result


# ─── One-Euro Filter ─────────────────────────────────────────────────────────
# Adaptive low-pass filter for motion capture smoothing.
# Ref: Casiez et al. "1€ Filter: A Simple Speed-based Low-pass Filter for
#       Noisy Input in Interactive Systems" (CHI 2012)
#
# Core idea: cutoff frequency increases with signal speed → low latency when
# moving fast, strong smoothing when stationary.

class OneEuroFilter1D:
    """One-Euro filter for a scalar signal."""

    def __init__(self, min_cutoff=1.0, beta=0.0, d_cutoff=1.0, dt=1.0/30):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.dt = dt
        self.x_prev = None
        self.dx_prev = 0.0

    @staticmethod
    def _smoothing_factor(t_e, cutoff):
        r = 2.0 * np.pi * cutoff * t_e
        return r / (r + 1.0)

    @staticmethod
    def _exponential_smoothing(a, x, x_prev):
        return a * x + (1.0 - a) * x_prev

    def __call__(self, x):
        if self.x_prev is None:
            self.x_prev = x
            return x

        # Derivative (speed estimate)
        t_e = self.dt
        a_d = self._smoothing_factor(t_e, self.d_cutoff)
        dx = (x - self.x_prev) / t_e
        dx_hat = self._exponential_smoothing(a_d, dx, self.dx_prev)

        # Adaptive cutoff
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)

        # Smooth
        a = self._smoothing_factor(t_e, cutoff)
        x_hat = self._exponential_smoothing(a, x, self.x_prev)

        self.x_prev = x_hat
        self.dx_prev = dx_hat
        return x_hat


def one_euro_filter_axis_angle(aa_sequence, detected_mask, fps=30,
                                min_cutoff=1.0, beta=0.007, d_cutoff=1.0):
    """
    Apply One-Euro filter to axis-angle parameters (N, 3).
    
    Each component is filtered independently. Gaps (detected=False) are
    handled by holding the last valid value without advancing the filter state.
    
    Args:
        aa_sequence: (N, 3) axis-angle array
        detected_mask: (N,) bool — which frames have valid data
        fps: frame rate
        min_cutoff: minimum cutoff frequency (Hz). Lower = more smoothing.
        beta: speed coefficient. Higher = less smoothing during fast motion.
        d_cutoff: derivative cutoff frequency (Hz)
    
    Returns:
        (N, 3) filtered axis-angle array
    """
    N = aa_sequence.shape[0]
    dt = 1.0 / fps
    result = aa_sequence.copy()

    # One filter per axis (3)
    filters = [OneEuroFilter1D(min_cutoff=min_cutoff, beta=beta,
                                d_cutoff=d_cutoff, dt=dt) for _ in range(3)]

    for f in range(N):
        if not detected_mask[f]:
            continue
        for ax in range(3):
            result[f, ax] = filters[ax](float(aa_sequence[f, ax]))

    return result


def one_euro_filter_hand_pose(hand_pose, detected_mask, fps=30,
                               min_cutoff=2.0, beta=0.01, d_cutoff=1.0):
    """
    Apply One-Euro filter to hand_pose (N, 45) — 15 joints × 3 axis-angle.
    
    Each of the 45 channels is filtered independently.
    
    Args:
        hand_pose: (N, 45) or (N, 15, 3) axis-angle
        detected_mask: (N,) bool
        fps: frame rate
        min_cutoff: cutoff for fingers (higher = less lag, less smooth)
        beta: speed coefficient
        d_cutoff: derivative cutoff
    
    Returns:
        (N, 45) filtered hand_pose
    """
    if hand_pose.ndim == 3:
        hp = hand_pose.reshape(hand_pose.shape[0], -1).copy()
    else:
        hp = hand_pose.copy()

    N, D = hp.shape
    dt = 1.0 / fps

    # One filter per channel
    filters = [OneEuroFilter1D(min_cutoff=min_cutoff, beta=beta,
                                d_cutoff=d_cutoff, dt=dt) for _ in range(D)]

    for f in range(N):
        if not detected_mask[f]:
            continue
        for ch in range(D):
            hp[f, ch] = filters[ch](float(hp[f, ch]))

    return hp


# ─── Butterworth low-pass filter (offline, zero-phase) ──────────────────────
def butterworth_filter_axis_angle(aa_sequence, detected_mask, fps=30,
                                   cutoff_hz=6.0, order=4):
    """
    Apply zero-phase Butterworth low-pass filter to axis-angle (N, 3).
    Uses scipy.signal.filtfilt for zero-phase delay (forward-backward filtering).
    
    Args:
        aa_sequence: (N, 3) axis-angle
        detected_mask: (N,) bool
        fps: frame rate
        cutoff_hz: cutoff frequency in Hz
        order: filter order
    
    Returns:
        (N, 3) filtered axis-angle
    """
    try:
        from scipy.signal import butter, filtfilt
    except ImportError:
        print("  WARNING: scipy not available, skipping Butterworth filter")
        return aa_sequence.copy()

    N = aa_sequence.shape[0]
    nyq = fps / 2.0
    if cutoff_hz >= nyq:
        return aa_sequence.copy()

    b, a = butter(order, cutoff_hz / nyq, btype='low')
    result = aa_sequence.copy()

    # Build valid segments and filter each
    for ax in range(3):
        signal = aa_sequence[:, ax].copy()
        # Fill gaps with linear interpolation for filtering
        valid_indices = np.where(detected_mask)[0]
        if len(valid_indices) < order * 2 + 1:
            continue
        # Interpolate gaps
        full_indices = np.arange(N)
        signal_interp = np.interp(full_indices, valid_indices, signal[valid_indices])
        # Apply filter
        result[:, ax] = filtfilt(b, a, signal_interp)

    # Restore non-detected frames to original
    for f in range(N):
        if not detected_mask[f]:
            result[f] = aa_sequence[f]

    return result


def butterworth_filter_hand_pose(hand_pose, detected_mask, fps=30,
                                  cutoff_hz=8.0, order=4):
    """
    Apply zero-phase Butterworth to hand_pose (N, 45).
    """
    try:
        from scipy.signal import butter, filtfilt
    except ImportError:
        print("  WARNING: scipy not available, skipping Butterworth filter")
        return hand_pose.copy()

    if hand_pose.ndim == 3:
        hp = hand_pose.reshape(hand_pose.shape[0], -1).copy()
    else:
        hp = hand_pose.copy()

    N, D = hp.shape
    nyq = fps / 2.0
    if cutoff_hz >= nyq:
        return hp

    b, a = butter(order, cutoff_hz / nyq, btype='low')
    valid_indices = np.where(detected_mask)[0]
    if len(valid_indices) < order * 2 + 1:
        return hp

    full_indices = np.arange(N)
    for ch in range(D):
        signal = hp[:, ch].copy()
        signal_interp = np.interp(full_indices, valid_indices, signal[valid_indices])
        hp[:, ch] = filtfilt(b, a, signal_interp)

    # Restore non-detected
    for f in range(N):
        if not detected_mask[f]:
            hp[f] = hand_pose[f] if hand_pose.ndim == 2 else hand_pose[f].reshape(-1)

    return hp


# ─── Anatomical wrist limits ─────────────────────────────────────────────────

# Maximum wrist rotation (Euler XYZ, degrees) relative to forearm.
# X: flexion/extension, Y: radial/ulnar deviation, Z: pronation/supination
WRIST_LIMITS_XYZ_DEG = np.array([45.0, 45.0, 90.0], dtype=np.float64)


def clamp_wrist_euler(R, limits_deg=WRIST_LIMITS_XYZ_DEG):
    """Clamp a wrist rotation matrix to anatomical limits.

    Decompose R into Euler XYZ angles, clamp each axis to ±limit,
    then reconstruct the rotation matrix.

    Args:
        R: (3,3) rotation matrix (local wrist rotation relative to forearm)
        limits_deg: (3,) max angle in degrees for [X, Y, Z]

    Returns:
        (3,3) clamped rotation matrix
    """
    limits_rad = np.deg2rad(limits_deg)

    # Rotation matrix -> Euler XYZ angles (intrinsic X->Y->Z order)
    # R = Rz @ Ry @ Rx  =>  column-vector convention
    # Extract Euler angles:
    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    singular = sy < 1e-6

    if not singular:
        x = np.arctan2(R[2, 1], R[2, 2])
        y = np.arctan2(-R[2, 0], sy)
        z = np.arctan2(R[1, 0], R[0, 0])
    else:
        x = np.arctan2(-R[1, 2], R[1, 1])
        y = np.arctan2(-R[2, 0], sy)
        z = 0.0

    # Clamp each axis
    x = np.clip(x, -limits_rad[0], limits_rad[0])
    y = np.clip(y, -limits_rad[1], limits_rad[1])
    z = np.clip(z, -limits_rad[2], limits_rad[2])

    # Reconstruct rotation matrix from clamped Euler angles: R = Rz @ Ry @ Rx
    cx, sx = np.cos(x), np.sin(x)
    cy, sy = np.cos(y), np.sin(y)
    cz, sz = np.cos(z), np.sin(z)

    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float64)
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float64)
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float64)

    return Rz @ Ry @ Rx


# ─── Forearm twist transfer ─────────────────────────────────────────────────

def forearm_twist_transfer(wrist_rotmats, body_pose_arr, global_orient_arr,
                            wrist_slice, elbow_slice, elbow_joint_idx,
                            detected, twist_share=0.55, twist_limit_deg=90.0):
    """Transfer part of wrist axial twist to the elbow (forearm pronation/supination).

    Decomposes each wrist rotation into swing + twist around the forearm axis.
    Regardless of the twist magnitude, a fraction (twist_share) of the total twist
    is transferred to the elbow via conjugation, which guarantees the combined
    elbow+wrist rotation is mathematically identical — no visual drift.

    Args:
        wrist_rotmats: list of (3,3) or None — wrist local rotations (modified in-place)
        body_pose_arr: (N, 63) — body_pose (elbow will be modified in-place)
        global_orient_arr: (N, 3) — global_orient for kinematic tree
        wrist_slice: slice for wrist in body_pose (e.g. slice(57,60))
        elbow_slice: slice for elbow in body_pose (e.g. slice(51,54))
        elbow_joint_idx: SMPL joint index for elbow (18=left, 19=right)
        detected: (N,) bool — which frames have valid hand detection
        twist_share: fraction of twist to transfer to elbow (0.0-1.0)
        twist_limit_deg: unused (kept for API compatibility)

    Returns:
        wrist_rotmats (modified in-place)
    """
    N = body_pose_arr.shape[0]

    # Forearm axis in SMPL rest pose (T-pose):
    #   Left:  wrist at +X from elbow → twist axis = +X
    #   Right: wrist at -X from elbow → twist axis = -X
    if wrist_slice == slice(57, 60):  # Left wrist
        twist_axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    else:  # Right wrist
        twist_axis = np.array([-1.0, 0.0, 0.0], dtype=np.float64)

    for f in range(N):
        if not detected[f] or wrist_rotmats[f] is None:
            continue

        R_wrist = wrist_rotmats[f]

        # Extract twist angle via swing-twist decomposition
        quat_wxyz = rotmat_to_quat(R_wrist)  # (w, x, y, z)
        w_q = quat_wxyz[0]
        vec = quat_wxyz[1:4]
        proj = twist_axis * np.dot(vec, twist_axis)
        twist_norm = np.sqrt(float(np.dot(proj, proj) + w_q * w_q))
        if twist_norm < 1e-8:
            continue  # No twist to transfer

        twist_quat = np.array([w_q / twist_norm, proj[0] / twist_norm,
                               proj[1] / twist_norm, proj[2] / twist_norm])
        if twist_quat[0] < 0:
            twist_quat = -twist_quat
        twist_angle = 2.0 * np.arctan2(float(np.dot(twist_quat[1:4], twist_axis)),
                                       float(twist_quat[0]))

        # Transfer a share of the twist to elbow via conjugation:
        #   R_wrist_new  = T^-1 @ R_wrist      (remove twist from wrist)
        #   R_elbow_new  = R_elbow @ T          (add twist to elbow)
        # This guarantees R_elbow_new @ R_wrist_new == R_elbow_old @ R_wrist_old
        transfer_angle = twist_angle * twist_share
        T = aa_to_rotmat(twist_axis * transfer_angle)

        # Wrist: left-multiply by T^-1
        wrist_rotmats[f] = T.T @ R_wrist

        # Elbow: right-multiply by T
        current_elbow_aa = body_pose_arr[f, elbow_slice].copy()
        current_elbow_R = aa_to_rotmat(current_elbow_aa)
        new_elbow_R = current_elbow_R @ T
        body_pose_arr[f, elbow_slice] = rotmat_to_aa(new_elbow_R)

    return wrist_rotmats


# ─── Pipeline entry point ────────────────────────────────────────────────────

def smooth_wrist_and_fingers(body_pose_arr, left_hand_pose, right_hand_pose,
                              left_wrist_rotmats, right_wrist_rotmats,
                              left_detected, right_detected,
                              fps=30, smooth_method='one_euro',
                              one_euro_wrist_min_cutoff=1.0,
                              one_euro_wrist_beta=0.007,
                              one_euro_finger_min_cutoff=2.0,
                              one_euro_finger_beta=0.01,
                              butter_wrist_cutoff_hz=6.0,
                              butter_finger_cutoff_hz=8.0,
                              max_jump_deg=90, smooth_fingers=True,
                              global_orient_arr=None,
                              forearm_twist=True,
                              forearm_twist_share=0.55,
                              forearm_twist_limit_deg=90.0):
    """
    Smooth wrist rotations and finger poses.

    Pipeline:
    0. Forearm twist transfer (move excess axial twist to elbow)
    1. Per-axis anatomical clamping
    2. Quaternion sign consistency (unwrap 180° flips)
    3. One-Euro or Butterworth filter on wrist axis-angle
    4. One-Euro or Butterworth filter on hand_pose
    5. Write results back

    Args:
        body_pose_arr: (N, 63) body_pose array — wrist at [57:60] and [60:63]
        left_hand_pose: (N, 45) left hand pose
        right_hand_pose: (N, 45) right hand pose
        left/right_wrist_rotmats: list of (3,3) or None per frame
        left/right_detected: (N,) bool
        fps: frame rate
        smooth_method: 'one_euro', 'butterworth', 'both', or 'none'
        max_jump_deg: max jump for initial quaternion unwrap (stage 2)
        global_orient_arr: (N, 3) global_orient — needed for forearm twist transfer
        forearm_twist: bool, enable forearm twist transfer
        forearm_twist_share: fraction of excess twist to transfer to elbow
        forearm_twist_limit_deg: twist beyond this is considered excess

    Returns:
        (body_pose_arr, left_hand_pose, right_hand_pose) — modified in-place
    """
    N = body_pose_arr.shape[0]

    for side, wrist_rotmats, detected, hp, wrist_slice, elbow_slice, elbow_jidx in [
        ('left',  left_wrist_rotmats,  left_detected,  left_hand_pose,
         slice(57, 60), slice(51, 54), 18),
        ('right', right_wrist_rotmats, right_detected, right_hand_pose,
         slice(60, 63), slice(54, 57), 19),
    ]:
        # --- Stage 0: Forearm twist transfer (BEFORE clamping) ---
        if forearm_twist and global_orient_arr is not None:
            forearm_twist_transfer(
                wrist_rotmats, body_pose_arr, global_orient_arr,
                wrist_slice=wrist_slice,
                elbow_slice=elbow_slice,
                elbow_joint_idx=elbow_jidx,
                detected=detected,
                twist_share=forearm_twist_share,
                twist_limit_deg=forearm_twist_limit_deg,
            )

        # --- Stage 1: Per-axis clamping ---
        clamped_rotmats = clamp_wrist_per_axis(wrist_rotmats,
                                                x_limit_deg=135, y_limit_deg=45, z_limit_deg=90)

        # --- Stage 2: Quaternion sign consistency (always) ---
        if max_jump_deg > 0:
            unwrapped = unwrap_wrist_rotations(clamped_rotmats, max_jump_deg=max_jump_deg)
        else:
            unwrapped = clamped_rotmats

        # Convert to axis-angle array
        wrist_aa = np.zeros((N, 3), dtype=np.float64)
        wrist_valid = np.zeros(N, dtype=bool)
        for f in range(N):
            if unwrapped[f] is not None:
                wrist_aa[f] = rotmat_to_aa(unwrapped[f])
                wrist_valid[f] = True

        # --- Stage 3: Smooth wrist ---
        if smooth_method in ('one_euro', 'both'):
            wrist_aa = one_euro_filter_axis_angle(
                wrist_aa, wrist_valid, fps=fps,
                min_cutoff=one_euro_wrist_min_cutoff,
                beta=one_euro_wrist_beta)
        if smooth_method in ('butterworth', 'both'):
            wrist_aa = butterworth_filter_axis_angle(
                wrist_aa, wrist_valid, fps=fps,
                cutoff_hz=butter_wrist_cutoff_hz)

        # Write wrist back
        for f in range(N):
            if unwrapped[f] is not None:
                body_pose_arr[f, wrist_slice] = wrist_aa[f]

        # --- Stage 4: Smooth fingers ---
        if smooth_fingers:
            if smooth_method in ('one_euro', 'both'):
                hp[:] = one_euro_filter_hand_pose(
                    hp, detected, fps=fps,
                    min_cutoff=one_euro_finger_min_cutoff,
                    beta=one_euro_finger_beta)
            if smooth_method in ('butterworth', 'both'):
                hp[:] = butterworth_filter_hand_pose(
                    hp, detected, fps=fps,
                    cutoff_hz=butter_finger_cutoff_hz)

    return body_pose_arr, left_hand_pose, right_hand_pose


def load_pkl(path):
    with open(path, 'rb') as f:
        try:
            return pickle.load(f)
        except pickle.UnpicklingError:
            pass
    # Fall back to torch.load for .pt files
    try:
        import torch
        return torch.load(path, map_location='cpu', weights_only=False)
    except Exception:
        pass
    raise RuntimeError(f"Cannot load {path}: neither pickle nor torch.format")


def save_pkl(data, path):
    with open(path, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


def aa_to_rotmat(aa):
    """Axis-angle (3,) -> rotation matrix (3, 3).
    Also accepts rotation matrix (3, 3) or (1, 3, 3) and returns it directly."""
    aa = np.asarray(aa, dtype=np.float64)
    # If already a rotation matrix, return as-is
    if aa.shape == (3, 3):
        return aa.copy()
    if aa.shape == (1, 3, 3):
        return aa[0].copy()
    if aa.shape != (3,):
        raise ValueError(f"aa_to_rotmat expects shape (3,), (3,3) or (1,3,3), got {aa.shape}")
    theta = np.linalg.norm(aa)
    if theta < 1e-10:
        return np.eye(3, dtype=np.float64)
    axis = aa / theta
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0],
    ], dtype=np.float64)
    return np.eye(3) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)


def rotmat_to_aa(R):
    """Rotation matrix (3, 3) -> axis-angle (3,)."""
    R = np.asarray(R, dtype=np.float64)
    cos_theta = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    theta = np.arccos(cos_theta)
    if theta < 1e-10:
        return np.zeros(3, dtype=np.float64)
    if np.pi - theta < 1e-5:
        # Near 180 degrees: use eigenvalue decomposition
        eigvals, eigvecs = np.linalg.eig(R)
        idx = np.argmin(np.abs(eigvals - 1.0))
        axis = np.real(eigvecs[:, idx])
        axis = axis / (np.linalg.norm(axis) + 1e-12)
        return axis * theta
    axis = np.array([
        R[2, 1] - R[1, 2],
        R[0, 2] - R[2, 0],
        R[1, 0] - R[0, 1],
    ]) / (2.0 * np.sin(theta))
    return axis * theta


def compute_global_rotation(full_body_pose, joint_idx):
    """
    Compute the global rotation of a joint by walking up the kinematic tree.
    
    Args:
        full_body_pose: (22, 3) axis-angle full body pose (global_orient + body_pose)
        joint_idx: joint index to compute global rotation for
    
    Returns:
        (3, 3) rotation matrix
    """
    global_rotation = np.eye(3)
    idx = joint_idx
    while idx != -1:
        joint_rot = aa_to_rotmat(full_body_pose[idx])
        global_rotation = joint_rot @ global_rotation
        idx = SMPL_PARENTS[idx]
    return global_rotation


def resample_hamer_data(hamer_data, src_frames, dst_frames,
                        src_fps=20.0, dst_fps=30.0):
    """Interpolate HAMER data to match GVHMR frame count using time-aligned mapping.

    Instead of uniform linspace mapping (frame number → frame number), this maps
    via real time: GVHMR frame g at time g/dst_fps should match HAMER frame
    g * src_fps / dst_fps.  This correctly handles non-integer fps ratios like
    27.26 → 10 where ffmpeg's fps=10 filter picks source frames at irregular
    frame-number intervals but regular time intervals.
    """
    from numpy import interp
    resampled = {}
    src_indices = np.arange(src_frames, dtype=np.float64)
    # Time-aligned: each GVHMR frame g → HAMER frame g * (hamer_fps / gvhmr_fps)
    dst_to_src = np.arange(dst_frames, dtype=np.float64) * src_fps / dst_fps

    for key, val in hamer_data.items():
        if isinstance(val, np.ndarray) and val.shape[0] == src_frames:
            if val.ndim == 1:
                resampled[key] = interp(dst_to_src, src_indices, val).astype(val.dtype)
            elif val.ndim == 2:
                out = np.zeros((dst_frames,) + val.shape[1:], dtype=val.dtype)
                for col in range(val.shape[1]):
                    out[:, col] = interp(dst_to_src, src_indices, val[:, col])
                resampled[key] = out
            elif val.ndim == 3:
                out = np.zeros((dst_frames,) + val.shape[1:], dtype=val.dtype)
                for c1 in range(val.shape[1]):
                    for c2 in range(val.shape[2]):
                        out[:, c1, c2] = interp(dst_to_src, src_indices, val[:, c1, c2])
                resampled[key] = out
            else:
                resampled[key] = val[:dst_frames] if dst_frames <= src_frames else val
        else:
            resampled[key] = val

    return resampled


def merge_hamer_into_gvhmr(gvhmr_data, hamer_data, resample=True, max_jump_deg=90,
                           smooth_method='one_euro',
                           one_euro_wrist_min_cutoff=1.0, one_euro_wrist_beta=0.007,
                           one_euro_finger_min_cutoff=2.0, one_euro_finger_beta=0.01,
                           butter_wrist_cutoff_hz=6.0, butter_finger_cutoff_hz=8.0,
                           wrist_mode='hamer', smooth_fingers=True,
                           forearm_twist=True,
                           hamer_fps=None, gvhmr_fps=None):
    """
    Merge HAMER hand poses into GVHMR body data using mano2smplx approach.
    
    For each frame and each SMPL param space (global/incam):
    1. Compute elbow global rotation from body kinematic tree
    2. HAMER global_orient = wrist rotation in hand-crop camera space (global)
    3. Apply Y-axis 180° correction (camera-to-body coordinate offset)
    4. Wrist local rotation = inv(elbow_global) @ wrist_global
    5. Left hand mirror: M @ R @ M where M = diag(-1, 1, 1)
    6. Finger poses: rotation matrices -> axis-angle (left mirror)
    """
    # Deep copy GVHMR data — convert torch tensors to numpy
    merged = {}
    for key in gvhmr_data:
        if isinstance(gvhmr_data[key], dict):
            merged[key] = {}
            for subkey in gvhmr_data[key]:
                v = gvhmr_data[key][subkey]
                if hasattr(v, 'cpu'):
                    v = v.cpu().numpy()
                merged[key][subkey] = v.copy() if hasattr(v, 'copy') else v
        else:
            v = gvhmr_data[key]
            if hasattr(v, 'cpu'):
                v = v.cpu().numpy()
            merged[key] = v.copy() if hasattr(v, 'copy') else v

    gvhmr_frames = gvhmr_data['smpl_params_global']['body_pose'].shape[0]
    hamer_frames = hamer_data['left_hand_pose'].shape[0]

    print(f"  GVHMR frames: {gvhmr_frames}")
    print(f"  HAMER frames: {hamer_frames}")

    # Determine actual fps for time-aligned resampling
    if gvhmr_fps is None:
        gvhmr_fps = float(gvhmr_data.get('fps', 30))
    if hamer_fps is None:
        # When HAMER runs on the same video as GVHMR, fps is the same
        hamer_fps = gvhmr_fps
    print(f"  GVHMR fps: {gvhmr_fps}, HAMER fps: {hamer_fps}")

    # Resample if needed
    if gvhmr_frames != hamer_frames:
        if resample and hamer_frames < gvhmr_frames:
            print(f"  Resampling HAMER {hamer_frames} -> {gvhmr_frames} frames (time-aligned)")
            hamer_data = resample_hamer_data(hamer_data, hamer_frames, gvhmr_frames,
                                             src_fps=hamer_fps, dst_fps=gvhmr_fps)
            num_frames = gvhmr_frames
        else:
            num_frames = min(gvhmr_frames, hamer_frames)
            print(f"  Using min frames: {num_frames}")
    else:
        num_frames = gvhmr_frames

    # Process each SMPL param space
    for space in ['smpl_params_global', 'smpl_params_incam']:
        if space not in merged:
            continue

        body_pose = merged[space]['body_pose']  # (N, 63)
        global_orient = merged[space]['global_orient']  # (N, 3)

        # Save original GVHMR wrist values (may be needed for gvhmr/differential modes)
        gvhmr_original_wrist = body_pose[:, 57:63].copy()

        # Add hand data from HAMER
        merged[space]['left_hand_pose'] = hamer_data['left_hand_pose'][:num_frames].copy()
        merged[space]['right_hand_pose'] = hamer_data['right_hand_pose'][:num_frames].copy()
        merged[space]['left_hand_global_orient'] = hamer_data['left_global_orient'][:num_frames].copy()
        merged[space]['right_hand_global_orient'] = hamer_data['right_global_orient'][:num_frames].copy()
        merged[space]['left_hand_betas'] = hamer_data['left_betas'][:num_frames].copy()
        merged[space]['right_hand_betas'] = hamer_data['right_betas'][:num_frames].copy()
        merged[space]['left_hand_is_detected'] = hamer_data['left_is_detected'][:num_frames].copy()
        merged[space]['right_hand_is_detected'] = hamer_data['right_is_detected'][:num_frames].copy()

        print(f"  Solving wrists (mano2smplx) in {space}...")

        # --- Pass 1: Compute wrist rotations for all frames ---
        left_wrist_rotmats = [None] * num_frames
        right_wrist_rotmats = [None] * num_frames
        left_elbow_globals = [None] * num_frames
        right_elbow_globals = [None] * num_frames

        for f in range(num_frames):
            # Build full body pose (22, 3): global_orient(1) + body_pose(21)
            full_pose = np.zeros((22, 3), dtype=np.float64)
            full_pose[0] = global_orient[f]
            full_pose[1:] = body_pose[f].reshape(21, 3)

            # Compute elbow global rotations
            left_elbow_globals[f] = compute_global_rotation(full_pose, 18)
            right_elbow_globals[f] = compute_global_rotation(full_pose, 19)

            # --- Left hand finger poses ---
            if hamer_data['left_is_detected'][f]:
                left_hp = np.asarray(hamer_data['left_hand_pose'][f], dtype=np.float64)
                if left_hp.shape == (45,):
                    left_hp = left_hp.reshape(15, 3)
                elif left_hp.shape == (15, 3, 3):
                    new_hp = np.zeros(45, dtype=np.float64)
                    for i in range(15):
                        finger_rot = M_MIRROR @ left_hp[i] @ M_MIRROR
                        new_hp[i*3:(i+1)*3] = rotmat_to_aa(finger_rot)
                    left_hp_flat = new_hp
                else:
                    left_hp_flat = left_hp.reshape(45)
                if left_hp.shape == (15, 3):
                    left_hp_flat = np.zeros(45, dtype=np.float64)
                    for i in range(15):
                        finger_rotmat = aa_to_rotmat(left_hp[i])
                        finger_rotmat = M_MIRROR @ finger_rotmat @ M_MIRROR
                        left_hp_flat[i*3:(i+1)*3] = rotmat_to_aa(finger_rotmat)
                merged[space]['left_hand_pose'][f] = left_hp_flat

                # Compute wrist rotation
                left_go = np.asarray(hamer_data['left_global_orient'][f], dtype=np.float64)
                left_wrist_global = aa_to_rotmat(left_go)
                left_wrist_global = M_MIRROR @ left_wrist_global @ M_MIRROR
                # Bind correction: MANO rest pose -> SMPL wrist rest pose
                left_wrist_global = RZ180 @ left_wrist_global
                left_wrist_local = np.linalg.inv(left_elbow_globals[f]) @ left_wrist_global
                # Anatomical limit: clamp wrist rotation to human range
                left_wrist_rotmats[f] = clamp_wrist_euler(left_wrist_local)

            # --- Right hand finger poses ---
            if hamer_data['right_is_detected'][f]:
                right_hp = np.asarray(hamer_data['right_hand_pose'][f], dtype=np.float64)
                if right_hp.shape == (45,):
                    right_hp_flat = right_hp
                elif right_hp.shape == (15, 3, 3):
                    right_hp_flat = np.zeros(45, dtype=np.float64)
                    for i in range(15):
                        right_hp_flat[i*3:(i+1)*3] = rotmat_to_aa(right_hp[i])
                else:
                    right_hp_flat = right_hp.reshape(45)
                merged[space]['right_hand_pose'][f] = right_hp_flat

                # Compute wrist rotation
                right_go = np.asarray(hamer_data['right_global_orient'][f], dtype=np.float64)
                right_wrist_global = aa_to_rotmat(right_go)
                # Bind correction: MANO rest pose -> SMPL wrist rest pose
                right_wrist_global = RZ180 @ right_wrist_global
                right_wrist_local = np.linalg.inv(right_elbow_globals[f]) @ right_wrist_global
                # Anatomical limit: clamp wrist rotation to human range
                right_wrist_rotmats[f] = clamp_wrist_euler(right_wrist_local)

        # --- Pass 1.5: Differential wrist mode ---
        # Instead of replacing GVHMR wrist with HAMER's (which mixes unstable
        # crop-camera space with body space), anchor HAMER's frame-to-frame
        # changes onto GVHMR's stable baseline.
        if wrist_mode == 'differential':
            for side_wrist_rotmats, wrist_slice, detected_arr in [
                (left_wrist_rotmats,  slice(57, 60), hamer_data['left_is_detected'][:num_frames]),
                (right_wrist_rotmats, slice(60, 63), hamer_data['right_is_detected'][:num_frames]),
            ]:
                # Find first detected frame as reference
                ref_f = -1
                for f in range(num_frames):
                    if detected_arr[f]:
                        ref_f = f
                        break
                if ref_f < 0 or side_wrist_rotmats[ref_f] is None:
                    # No detection at all, skip
                    continue

                # GVHMR's original wrist rotation at reference frame
                gvhmr_ref_aa = body_pose[ref_f, wrist_slice].copy()
                gvhmr_ref_R = aa_to_rotmat(gvhmr_ref_aa)

                # HAMER's computed wrist at reference frame (already local: inv(elbow)@wrist_global)
                hamer_ref_R = side_wrist_rotmats[ref_f]

                # Calibration offset: HAMER→GVHMR at reference frame
                # offset @ hamer = gvhmr  =>  offset = gvhmr @ inv(hamer)
                offset_R = gvhmr_ref_R @ np.linalg.inv(hamer_ref_R)

                # Apply offset to all frames: anchor HAMER changes to GVHMR space
                for f in range(num_frames):
                    if side_wrist_rotmats[f] is not None:
                        side_wrist_rotmats[f] = offset_R @ side_wrist_rotmats[f]

            print(f"    Wrist mode: differential (HAMER changes anchored to GVHMR baseline)")

        # --- Pass 2 & 3: Smooth and write back ---
        fps = float(merged.get('fps', gvhmr_fps))  # Use pkl fps for smoothing (params tuned at 30fps)
        smooth_wrist_and_fingers(
            body_pose,
            merged[space]['left_hand_pose'],
            merged[space]['right_hand_pose'],
            left_wrist_rotmats, right_wrist_rotmats,
            hamer_data['left_is_detected'][:num_frames],
            hamer_data['right_is_detected'][:num_frames],
            fps=fps, smooth_method=smooth_method,
            one_euro_wrist_min_cutoff=one_euro_wrist_min_cutoff,
            one_euro_wrist_beta=one_euro_wrist_beta,
            one_euro_finger_min_cutoff=one_euro_finger_min_cutoff,
            one_euro_finger_beta=one_euro_finger_beta,
            butter_wrist_cutoff_hz=butter_wrist_cutoff_hz,
            butter_finger_cutoff_hz=butter_finger_cutoff_hz,
            max_jump_deg=max_jump_deg,
            smooth_fingers=smooth_fingers,
            global_orient_arr=global_orient,
            forearm_twist=forearm_twist,
        )

        # gvhmr mode: restore original GVHMR wrist (ignore HAMER wrist entirely)
        if wrist_mode == 'gvhmr':
            body_pose[:, 57:63] = gvhmr_original_wrist
            print(f"    Wrist mode: gvhmr (kept GVHMR original wrist)")

    # Trim frames if HAMER had fewer
    if num_frames < gvhmr_frames:
        for space in ['smpl_params_global', 'smpl_params_incam']:
            if space not in merged:
                continue
            for key in list(merged[space].keys()):
                if isinstance(merged[space][key], np.ndarray) and merged[space][key].shape[0] > num_frames:
                    merged[space][key] = merged[space][key][:num_frames]

    merged['hamer_merge_info'] = {
        'source': 'HAMER + GVHMR merge (mano2smplx approach)',
        'method': 'mano2smplx',
        'wrist_mode': wrist_mode,
        'smooth_method': smooth_method,
        'gvhmr_frames': int(gvhmr_frames),
        'hamer_frames_original': int(hamer_frames),
        'merged_frames': int(num_frames),
        'resampled': bool(resample and hamer_frames != gvhmr_frames and hamer_frames < gvhmr_frames),
        'left_hand_detected_frames': int(hamer_data['left_is_detected'][:num_frames].sum()),
        'right_hand_detected_frames': int(hamer_data['right_is_detected'][:num_frames].sum()),
        'wrist_unwrap_max_jump_deg': float(max_jump_deg),
        'one_euro_wrist_min_cutoff': float(one_euro_wrist_min_cutoff),
        'one_euro_wrist_beta': float(one_euro_wrist_beta),
        'one_euro_finger_min_cutoff': float(one_euro_finger_min_cutoff),
        'one_euro_finger_beta': float(one_euro_finger_beta),
    }

    return merged


def main():
    parser = argparse.ArgumentParser(
        description='Merge HAMER hand data into GVHMR body data (mano2smplx approach)')
    parser.add_argument('--gvhmr_pkl', type=str, required=True,
                        help='Path to GVHMR pkl file')
    parser.add_argument('--hamer_pkl', type=str, required=True,
                        help='Path to HAMER hand_data.pkl (dict format)')
    parser.add_argument('--output', type=str,
                        help='Output merged pkl path')
    parser.add_argument('--no_resample', action='store_true',
                        help='Disable HAMER frame resampling')
    parser.add_argument('--no_finger_smooth', action='store_true',
                        help='Disable finger smoothing')
    parser.add_argument('--no_forearm_twist', action='store_true',
                        help='Disable forearm twist transfer (wrist axial twist stays at wrist)')
    parser.add_argument('--max_jump_deg', type=float, default=90,
                        help='Max per-frame wrist rotation change in degrees (default: 90, 0=disable unwrap)')
    parser.add_argument('--smooth', type=str, default='one_euro',
                        choices=['one_euro', 'butterworth', 'both', 'none'],
                        help='Smoothing method (default: one_euro)')
    parser.add_argument('--wrist_mode', type=str, default='hamer',
                        choices=['hamer', 'gvhmr', 'differential'],
                        help='Wrist rotation source: hamer=HAMER computed, gvhmr=keep GVHMR original, '
                             'differential=HAMER frame-to-frame changes anchored to GVHMR baseline (default: hamer)')
    # One-Euro parameters
    oe = parser.add_argument_group('One-Euro filter')
    oe.add_argument('--wrist_cutoff', type=float, default=1.0,
                    help='One-Euro wrist min cutoff Hz (lower=smoother, default: 1.0)')
    oe.add_argument('--wrist_beta', type=float, default=0.007,
                    help='One-Euro wrist beta (higher=less smoothing when fast, default: 0.007)')
    oe.add_argument('--finger_cutoff', type=float, default=2.0,
                    help='One-Euro finger min cutoff Hz (default: 2.0)')
    oe.add_argument('--finger_beta', type=float, default=0.01,
                    help='One-Euro finger beta (default: 0.01)')
    # Butterworth parameters
    bw = parser.add_argument_group('Butterworth filter')
    bw.add_argument('--butter_wrist_hz', type=float, default=6.0,
                    help='Butterworth wrist cutoff Hz (default: 6.0)')
    bw.add_argument('--butter_finger_hz', type=float, default=8.0,
                    help='Butterworth finger cutoff Hz (default: 8.0)')

    # FPS parameters for time-aligned resampling
    parser.add_argument('--hamer_fps', type=float, default=None,
                        help='FPS of the video fed to HAMER (default: None, auto-detect from frame count / gvhmr_fps)')
    parser.add_argument('--gvhmr_fps', type=float, default=None,
                        help='FPS of the original video fed to GVHMR (default: read from pkl or 30)')

    args = parser.parse_args()

    if not os.path.exists(args.gvhmr_pkl):
        print(f"ERROR: GVHMR pkl not found: {args.gvhmr_pkl}")
        return 1
    if not os.path.exists(args.hamer_pkl):
        print(f"ERROR: HAMER pkl not found: {args.hamer_pkl}")
        return 1

    # Default output path
    if not args.output:
        from pathlib import Path
        gvhmr_path = Path(args.gvhmr_pkl)
        args.output = str(gvhmr_path.parent / 'merged_mano2smplx.pkl')

    print(f"Loading GVHMR: {args.gvhmr_pkl}")
    gvhmr_data = load_pkl(args.gvhmr_pkl)

    print(f"Loading HAMER: {args.hamer_pkl}")
    hamer_data = load_pkl(args.hamer_pkl)

    # Verify HAMER format — auto-convert Stitch format if needed
    if not isinstance(hamer_data, dict):
        print(f"ERROR: HAMER pkl is not a dict (got {type(hamer_data)})")
        return 1

    # Stitch format: {global_orient:(N,2,3,3), hand_pose:(N,2,15,3,3), global_orient_valid:(N,2)}
    # Our format: {left_hand_pose:(N,45), right_hand_pose:(N,45), left_global_orient:(N,3), ...}
    if 'global_orient' in hamer_data and 'hand_pose' in hamer_data and 'left_hand_pose' not in hamer_data:
        import numpy as np
        print("  Converting Stitch HAMER format to mano2smplx format...")
        go = hamer_data['global_orient']      # (N, 2, 3, 3) rotation matrices
        hp = hamer_data['hand_pose']           # (N, 2, 15, 3, 3) rotation matrices
        valid = hamer_data.get('global_orient_valid', np.ones((go.shape[0], 2), dtype=bool))

        N = go.shape[0]
        # Convert global_orient rotmats to axis-angle
        left_go_aa = np.zeros((N, 3), dtype=np.float64)
        right_go_aa = np.zeros((N, 3), dtype=np.float64)
        for i in range(N):
            left_go_aa[i] = rotmat_to_aa(go[i, 0].astype(np.float64))
            right_go_aa[i] = rotmat_to_aa(go[i, 1].astype(np.float64))

        # Convert hand_pose rotmats to axis-angle: (N, 2, 15, 3, 3) -> (N, 45)
        left_hp_aa = np.zeros((N, 45), dtype=np.float64)
        right_hp_aa = np.zeros((N, 45), dtype=np.float64)
        for i in range(N):
            for j in range(15):
                left_hp_aa[i, j*3:(j+1)*3] = rotmat_to_aa(hp[i, 0, j].astype(np.float64))
                right_hp_aa[i, j*3:(j+1)*3] = rotmat_to_aa(hp[i, 1, j].astype(np.float64))

        hamer_data = {
            'left_global_orient': left_go_aa,
            'right_global_orient': right_go_aa,
            'left_hand_pose': left_hp_aa,
            'right_hand_pose': right_hp_aa,
            'left_is_detected': valid[:, 0].astype(bool),
            'right_is_detected': valid[:, 1].astype(bool),
        }
        # Provide default betas
        hamer_data['left_betas'] = np.zeros((N, 10), dtype=np.float32)
        hamer_data['right_betas'] = np.zeros((N, 10), dtype=np.float32)
        print(f"  Converted: N={N}, left_detected={hamer_data['left_is_detected'].sum()}, right_detected={hamer_data['right_is_detected'].sum()}")

    required_keys = ['left_hand_pose', 'right_hand_pose', 'left_global_orient', 'right_global_orient']
    for key in required_keys:
        if key not in hamer_data:
            print(f"ERROR: Missing key in HAMER pkl: {key}")
            return 1

    print("Merging...")
    merged = merge_hamer_into_gvhmr(
        gvhmr_data, hamer_data,
        resample=not args.no_resample,
        max_jump_deg=args.max_jump_deg,
        smooth_method=args.smooth,
        wrist_mode=args.wrist_mode,
        one_euro_wrist_min_cutoff=args.wrist_cutoff,
        one_euro_wrist_beta=args.wrist_beta,
        one_euro_finger_min_cutoff=args.finger_cutoff,
        one_euro_finger_beta=args.finger_beta,
        butter_wrist_cutoff_hz=args.butter_wrist_hz,
        butter_finger_cutoff_hz=args.butter_finger_hz,
        smooth_fingers=not args.no_finger_smooth,
        forearm_twist=not args.no_forearm_twist,
        hamer_fps=args.hamer_fps,
        gvhmr_fps=args.gvhmr_fps,
    )

    save_pkl(merged, args.output)

    info = merged['hamer_merge_info']
    print(f"\nDone! Saved to: {args.output}")
    print(f"  Total frames: {info['merged_frames']}")
    print(f"  Left hand detected: {info['left_hand_detected_frames']}/{info['merged_frames']}")
    print(f"  Right hand detected: {info['right_hand_detected_frames']}/{info['merged_frames']}")
    print(f"  Method: mano2smplx (elbow kinematic chain)")
    print(f"  Smoothing: {info.get('smooth_method', 'none')}")
    if info.get('smooth_method') == 'one_euro':
        print(f"    Wrist: cutoff={info.get('one_euro_wrist_min_cutoff')} Hz, beta={info.get('one_euro_wrist_beta')}")
        print(f"    Finger: cutoff={info.get('one_euro_finger_min_cutoff')} Hz, beta={info.get('one_euro_finger_beta')}")
    print(f"  Quaternion unwrap: {info['wrist_unwrap_max_jump_deg']}° max jump")

    return 0


if __name__ == '__main__':
    exit(main())
