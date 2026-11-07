"""Finalise the hand-mean execution: FBX validation + sampling-seed stability.

CPU only. The seed-stability check answers whether the N=60/N=120 quality drop is a
property of the sampling budget or just of the one seed we happened to draw.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

P = Path(__file__).resolve().parents[1]
REPO = P / "app" / "flowhmr"
sys.path.insert(0, str(REPO))
OUT = P / "outputs" / "hand_mean_modes"
T = 701
SEED = 20261014


def sampled_indices(n_frames: int, n: int, seed: int) -> np.ndarray:
    rng = np.random.RandomState(seed)
    picked = []
    for start in range(0, n_frames, n):
        end = min(start + n, n_frames)
        picked.append(int(rng.randint(start, end)))
    return np.array(sorted(set(picked)), dtype=int)


def main() -> int:
    import fbx

    sys.path.insert(0, str(REPO))
    from flowhmr.core.export.fbx_utils import SMPLH_JOINT2NUM

    validation = []
    for name in ("perframe", "mean_full", "mean_n60", "mean_n120"):
        fbx_path = OUT / f"HAND_{name}.fbx"
        npz_path = OUT / f"HAND_{name}.npz"
        with np.load(npz_path, allow_pickle=False) as z:
            poses = z["poses"]
        m = fbx.FbxManager.Create()
        m.SetIOSettings(fbx.FbxIOSettings.Create(m, fbx.IOSROOT))
        imp = fbx.FbxImporter.Create(m, "")
        assert imp.Initialize(str(fbx_path), -1, m.GetIOSettings()), imp.GetStatus().GetErrorString()
        scene = fbx.FbxScene.Create(m, "check")
        assert imp.Import(scene)
        imp.Destroy()
        nodes, meshes = [], []

        def walk(node):
            attr = node.GetNodeAttribute()
            if attr and attr.GetAttributeType() == fbx.FbxNodeAttribute.EType.eSkeleton:
                nodes.append(node.GetName())
            if attr and attr.GetAttributeType() == fbx.FbxNodeAttribute.EType.eMesh:
                meshes.append(node)
            for i in range(node.GetChildCount()):
                walk(node.GetChild(i))

        walk(scene.GetRootNode())
        assert len(nodes) == 52 and len(meshes) == 1, (name, len(nodes), len(meshes))
        m.Destroy()
        validation.append({"mode": name, "frames": int(len(poses)), "skeletons": len(nodes),
                           "meshes": len(meshes), "fbx_bytes": fbx_path.stat().st_size})

    # Sampling-seed stability of the sparse mean estimate (no model runs).
    feat = torch.load(P / "outputs" / "sam_hand_modes" / "feat_full.pt",
                      map_location="cpu", weights_only=True)
    hands = feat[:, 1024:]
    left, right = hands[:, :1024], hands[:, 1024:]
    full_l = left.mean(dim=0, keepdim=True)
    full_r = right.mean(dim=0, keepdim=True)
    seeds = [SEED, 1, 2, 3, 4]
    stability = {}
    for n in (60, 120):
        dl, dr = [], []
        for s in seeds:
            idx = sampled_indices(T, n, s)
            dl.append(float((left[idx].mean(dim=0, keepdim=True) - full_l).abs().max()))
            dr.append(float((right[idx].mean(dim=0, keepdim=True) - full_r).abs().max()))
        stability[f"n{n}"] = {
            "seeds": seeds,
            "left_max_abs_error_vs_full_mean": dl,
            "right_max_abs_error_vs_full_mean": dr,
            "left_mean_of_errors": float(np.mean(dl)),
            "left_worst_of_errors": float(np.max(dl)),
            "samples_per_seed": int(sampled_indices(T, n, SEED).size),
        }
    per_frame_spread = float((hands - hands.mean(dim=0, keepdim=True)).norm(dim=1).mean())

    report = {
        "fbx_validation": validation,
        "sampling_seed_stability": stability,
        "per_frame_hand_deviation_from_mean_norm": per_frame_spread,
        "interpretation": (
            "Compare the seed spread against the per-frame hand deviation (~22) and against "
            "the fixed-mean condition error: a small seed spread means the sampling budget "
            "itself is adequate and the quality drop comes from dropping per-frame hand timing."
        ),
    }
    path = OUT / "final_checks.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
