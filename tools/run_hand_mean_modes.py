"""Execute SPARSE_MEAN_HAND_PLAN.md phase 1+2 (quality).

Body features are identical in every mode (taken from the same local cache), so any
difference is caused only by how the left/right hand reference information is built:

  perframe     : hand tokens every frame (baseline)
  mean_full    : one fixed L/R mean over the whole valid range (the author's approach)
  mean_n60     : one fixed L/R mean estimated from ~T/60 sampled frames
  mean_n120    : same with N=120

Sampling rule (plan item 3): split into segments of N frames and pick ONE frame per
segment with a fixed seed, so a periodic dance cannot always land on the same beat.
Non-finite rows are dropped; the sampling indices are written out for reproducibility.

Same camera / global noise / weights / steps / postprocess / HaMeR fingers in all modes.
No production file is modified.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

P = Path(__file__).resolve().parents[1]
TOOLS = P / "tools"
sys.path.insert(0, str(TOOLS))

spec = importlib.util.spec_from_file_location("rsm", TOOLS / "run_sam_hand_modes.py")
rsm = importlib.util.module_from_spec(spec)
sys.modules["rsm"] = rsm
spec.loader.exec_module(rsm)   # env + sys.path only; __main__ guard prevents a run

from flowhmr.core.export.smplh2fbx import SMPLH2FBX  # noqa: E402
from flowhmr.core.math.geometry import (  # noqa: E402
    rotation_matrix_to_angle_axis,
    rot6d_to_rotation_matrix,
)
from flowhmr.pipeline.v2m_inference import (  # noqa: E402
    V2MInferenceEngine,
    _rotate_smplh_params_y180,
    build_feature_dict,
)

OUT = P / "outputs" / "hand_mean_modes"
OUT.mkdir(parents=True, exist_ok=True)
REPORT = OUT / "quality_report.json"
SEED = 20261014
F0, F1 = rsm.F0, rsm.F1
T = F1 - F0 + 1
BASE_FEAT = P / "outputs" / "sam_hand_modes" / "feat_full.pt"
VIOLENT_1BASED = (928, 1370)


def sampled_indices(n_frames: int, n: int, seed: int) -> np.ndarray:
    rng = np.random.RandomState(seed)
    picked = []
    for start in range(0, n_frames, n):
        end = min(start + n, n_frames)
        picked.append(int(rng.randint(start, end)))
    return np.array(sorted(set(picked)), dtype=int)


def main() -> int:
    feat = torch.load(BASE_FEAT, map_location="cpu", weights_only=True)
    assert feat.shape == (T, 3072), feat.shape
    body = feat[:, :1024]
    hands = feat[:, 1024:]                      # 2048 = left 1024 + right 1024
    finite = torch.isfinite(feat).all(dim=1).numpy()
    valid = np.flatnonzero(finite)
    print(f"[setup] frames={T} valid={valid.size}", flush=True)

    left = hands[:, :1024]
    right = hands[:, 1024:]
    mean_left = left[valid].mean(dim=0, keepdim=True)
    mean_right = right[valid].mean(dim=0, keepdim=True)
    print(f"[mean] per-channel |mean| avg = {mean_left.abs().mean():.4f}/"
          f"{mean_right.abs().mean():.4f}", flush=True)

    variants = {"perframe": hands.clone()}
    variants["mean_full"] = torch.cat([mean_left.expand(T, -1), mean_right.expand(T, -1)], dim=1)
    indices = {}
    for n in (60, 120):
        idx = np.intersect1d(sampled_indices(T, n, SEED), valid)
        indices[f"mean_n{n}"] = idx.tolist()
        ml = left[idx].mean(dim=0, keepdim=True)
        mr = right[idx].mean(dim=0, keepdim=True)
        variants[f"mean_n{n}"] = torch.cat([ml.expand(T, -1), mr.expand(T, -1)], dim=1)
        print(f"[mean] N={n}: {idx.size} samples (indices {idx[:6].tolist()}...)", flush=True)

    cam_raw = np.load(rsm.CACHE / "video_cut_1080p30_vggt_camera.npz")["RT_vggt"].astype(np.float32)[F0:F1 + 1]
    cam = rsm._vggt_opencv_to_yup_rt(torch.from_numpy(cam_raw))
    noise = torch.load(rsm.CACHE / "global_noise_seed0.pt", map_location="cpu", weights_only=True)[:, F0:F1 + 1]
    with np.load(rsm.REF, allow_pickle=False) as z:
        ref = {k: z[k].copy() for k in z.files}

    # Feature diagnostics: how far is each fixed-mean condition from the true per-frame one,
    # and how far is the sparse estimate from the full-range mean?
    diag = {}
    base_lr = torch.cat([mean_left.expand(T, -1), mean_right.expand(T, -1)], dim=1)
    for name, h in variants.items():
        d = (h - hands).norm(dim=1)
        diag[name] = {
            "hand_condition_diff_vs_perframe_mean": float(d.mean()),
            "hand_condition_diff_vs_perframe_max": float(d.max()),
            "diff_vs_full_mean_condition_mean": float((h - base_lr).norm(dim=1).mean()),
        }
    print("[diag] " + json.dumps({k: round(v["hand_condition_diff_vs_perframe_mean"], 3)
                                  for k, v in diag.items()}), flush=True)

    engine = V2MInferenceEngine.build_lite(
        str(rsm.REPO / "checkpoints/flowhmr_latest/flowhmr_latest.ckpt"),
        str(rsm.CACHE / "original_checkpoint_config.yml"),
        device="cuda", postprocess=True,
    )
    original_randn = rsm.pm.randn_tensor
    report = {
        "plan": "logs/SPARSE_MEAN_HAND_PLAN.md",
        "frames_1based": [F0 + 1, F1 + 1],
        "violent_1based": list(VIOLENT_1BASED),
        "sampling_seed": SEED,
        "sampling_rule": "segments of N frames, one fixed-seed random frame per segment",
        "sampled_indices_global_0based": indices,
        "feature_diagnostics": diag,
        "modes": {},
        "exports": [],
        "notes": [
            "Body features identical across modes; only the hand reference changes.",
            "Mean hand condition is built from the SAME local cache range for every variant.",
            "L and R means computed separately (the author's wording was ambiguous).",
            "Fingers copied from the user-specified HaMeR reference; hands are not judged here.",
            "No production file changed.",
        ],
    }

    for name in ("perframe", "mean_full", "mean_n60", "mean_n120"):
        h = variants[name]
        feats = torch.cat([body, h], dim=1).to(torch.float32).contiguous()
        calls = []

        def fixed(shape, generator=None, device=None, dtype=None, layout=None, _c=calls):
            assert tuple(shape) == (1, T, 491), (shape, T)
            _c.append(1)
            return noise.clone().to(device=device or "cpu", dtype=dtype or torch.float32)

        rsm.pm.randn_tensor = fixed
        try:
            torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()
            fdict = build_feature_dict(feats, cam, engine.device)
            with torch.no_grad():
                out = engine.pipeline.generate(feature=fdict, seeds=[0], length=T, cfg_scale=1.0)
            assert calls == [1], calls
            gen_s = time.perf_counter() - t0
            t1 = time.perf_counter()
            out = engine.postprocess.run(out)
            post_s = time.perf_counter() - t1
            poses = rotation_matrix_to_angle_axis(
                rot6d_to_rotation_matrix(out["rot6d"][0])).detach().cpu().numpy().copy()
            trans = out["trans"][0].detach().cpu().numpy().copy()
            _rotate_smplh_params_y180(poses, trans)
            poses[:, 22:] = ref["poses"][F0:F1 + 1, 22:]
            npz = OUT / f"HAND_{name}.npz"
            fbx = OUT / f"HAND_{name}.fbx"
            np.savez(npz, poses=poses.astype(np.float32), trans=trans.astype(np.float32),
                     betas=ref["betas"], Rh=poses[:, 0].astype(np.float32),
                     mocap_framerate=30, num_frames=T, gender="neutral")
            conv = SMPLH2FBX(str(rsm.REPO / "assets/body_models/smplh/neutral/model.npz"),
                             obj_template_path=None)
            assert conv.convert_npz_to_fbx(str(npz), str(fbx))
            conv.mgr.Destroy()
            report["modes"][name] = {
                "generate_seconds": gen_s, "postprocess_seconds": post_s,
                "peak_alloc_GiB": torch.cuda.max_memory_allocated() / 2**30,
            }
            report["exports"].append({"mode": name, "npz": str(npz), "fbx": str(fbx),
                                      "bytes": fbx.stat().st_size})
            print(f"[gen] {name}: generate={gen_s:.1f}s postprocess={post_s:.1f}s", flush=True)
            del out, fdict
            torch.cuda.empty_cache()
        finally:
            rsm.pm.randn_tensor = original_randn

    # Body metrics versus the per-frame baseline.
    from scipy.spatial.transform import Rotation as Rot

    def geo(a, b):
        return np.degrees((Rot.from_rotvec(a.reshape(-1, 3)).inv()
                           * Rot.from_rotvec(b.reshape(-1, 3))).magnitude()).reshape(-1, 52)

    def st(v):
        return {"mean": float(v.mean()), "p99": float(np.percentile(v, 99)), "max": float(v.max())}

    with np.load(OUT / "HAND_perframe.npz") as z:
        base = {k: z[k].copy() for k in z.files}
    vi = slice(VIOLENT_1BASED[0] - F0 - 1, VIOLENT_1BASED[1] - F0)
    for name in ("mean_full", "mean_n60", "mean_n120"):
        with np.load(OUT / f"HAND_{name}.npz") as z:
            cur = {k: z[k].copy() for k in z.files}
        assert np.array_equal(cur["poses"][:, 22:], base["poses"][:, 22:]), "fingers must match"
        e = geo(base["poses"], cur["poses"])
        dt = np.linalg.norm(base["trans"] - cur["trans"], axis=1) * 100
        dta = np.linalg.norm((base["trans"] - base["trans"].mean(0))
                             - (cur["trans"] - cur["trans"].mean(0)), axis=1) * 100
        report["modes"][name].update({
            "body_deg": st(e[:, 1:20]), "root_deg": st(e[:, 0]),
            "trans_cm_absolute": st(dt), "trans_cm_mean_removed": st(dta),
            "violent_body_deg": st(e[vi][:, 1:20]), "violent_trans_cm": st(dt[vi]),
        })
        print(f"[metric] {name} body={json.dumps(report['modes'][name]['body_deg'])} "
              f"trans={report['modes'][name]['trans_cm_absolute']['mean']:.2f}cm", flush=True)

    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[done] " + str(REPORT), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
