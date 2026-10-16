import pickle, numpy as np, sys
pkl_path = sys.argv[1]
with open(pkl_path, 'rb') as f:
    results = pickle.load(f)
if isinstance(results, dict):
    print("Already dict format, skip conversion")
    sys.exit(0)
converted = {
    'left_hand_pose': np.zeros((1, 45), dtype=np.float32),
    'right_hand_pose': np.zeros((1, 45), dtype=np.float32),
    'left_global_orient': np.zeros((1, 3), dtype=np.float32),
    'right_global_orient': np.zeros((1, 3), dtype=np.float32),
    'left_betas': np.zeros((1, 10), dtype=np.float32),
    'right_betas': np.zeros((1, 10), dtype=np.float32),
    'left_is_detected': np.array([False]),
    'right_is_detected': np.array([False]),
}
if isinstance(results, list) and len(results) == 0:
    with open(pkl_path, 'wb') as f:
        pickle.dump(converted, f, protocol=pickle.HIGHEST_PROTOCOL)
    print("No HAMER hands detected; wrote empty hand dict")
    sys.exit(0)
if not isinstance(results, list):
    print("ERROR: unexpected format")
    sys.exit(1)
# Debug: print actual mano_params keys and shapes
for i, r in enumerate(results):
    mp = r.get('mano_params', {})
    print(f"  Hand {i}: is_right={r.get('is_right')}, mano_params keys={list(mp.keys()) if isinstance(mp, dict) else type(mp)}")
    if isinstance(mp, dict):
        for k, v in mp.items():
            print(f"    {k}: shape={getattr(v, 'shape', type(v))}")
for r in results:
    is_right = bool(r.get('is_right', 0))
    prefix = 'right' if is_right else 'left'
    mp = r.get('mano_params', {})
    if isinstance(mp, dict):
        for hp_key in ('right_hand_pose', 'hand_pose'):
            hp = mp.get(hp_key)
            if hp is not None:
                # If rotation matrices (N,3,3), convert to axis-angle
                if hp.ndim == 3 and hp.shape[1:] == (3, 3):
                    from scipy.spatial.transform import Rotation
                    hp = Rotation.from_matrix(hp).as_rotvec().reshape(-1, 45)
                    print(f"  Converted rotation matrices to axis-angle: {hp.shape}")
                else:
                    hp = np.atleast_2d(hp)
                    if hp.shape[1] < 45:
                        padded = np.zeros((hp.shape[0], 45), dtype=hp.dtype)
                        padded[:, :hp.shape[1]] = hp
                        hp = padded
                converted[prefix + '_hand_pose'] = hp[:, :45]
                print(f"  {prefix}_hand_pose from '{hp_key}': {hp.shape}")
                break
        for go_key in ('global_orient', 'right_hand_global_orient'):
            go = mp.get(go_key)
            if go is not None:
                # If rotation matrix (N,3,3) or (3,3), convert to axis-angle (N,3)
                if go.ndim == 3 and go.shape[1:] == (3, 3):
                    from scipy.spatial.transform import Rotation
                    go = Rotation.from_matrix(go).as_rotvec()
                    print(f"  Converted global_orient rotation matrices to axis-angle: {go.shape}")
                elif go.ndim == 2 and go.shape == (3, 3):
                    from scipy.spatial.transform import Rotation
                    go = Rotation.from_matrix(go.reshape(1, 3, 3)).as_rotvec().reshape(1, 3)
                    print(f"  Converted global_orient 3x3 matrix to axis-angle: {go.shape}")
                converted[prefix + '_global_orient'] = np.atleast_2d(go)[:, :3]
                break
        for bt_key in ('betas', 'right_hand_betas'):
            bt = mp.get(bt_key)
            if bt is not None:
                converted[prefix + '_betas'] = np.atleast_2d(bt)[:, :10]
                break
    converted[prefix + '_is_detected'] = np.array([True])
with open(pkl_path, 'wb') as f:
    pickle.dump(converted, f, protocol=pickle.HIGHEST_PROTOCOL)
print("Converted HAMER image output to dict format")
