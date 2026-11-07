"""Execute SPARSE_MEAN_HAND_PLAN.md cost clause.

Same session, same frames, four real SAM schedules:
  full        : hands every frame (original)
  sparse_n60  : body every frame + hands only on the seeded sampled frames
  sparse_n120 : same with N=120
  body_only   : body every frame, no hands at all (floor)

Timing includes the sampling forwards, the per-frame body forwards, the mean
accumulation and the frame decoding, i.e. the real cost of obtaining the mean.
Body tokens are verified against the cached baseline so the quality run stays valid.
Read-only w.r.t. production code.
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
spec.loader.exec_module(rsm)

import cv2  # noqa: E402

OUT = P / "outputs" / "hand_mean_modes"
OUT.mkdir(parents=True, exist_ok=True)
REPORT = OUT / "cost_report.json"
SEED = 20261014
F0, F1 = rsm.F0, rsm.F1
T = F1 - F0 + 1
SCHEDULES = [("full", None), ("sparse_n60", 60), ("sparse_n120", 120), ("body_only", 0)]


def sampled_indices(n_frames: int, n: int, seed: int) -> np.ndarray:
    rng = np.random.RandomState(seed)
    picked = []
    for start in range(0, n_frames, n):
        end = min(start + n, n_frames)
        picked.append(int(rng.randint(start, end)))
    return np.array(sorted(set(picked)), dtype=int)


def main() -> int:
    bbox = np.load(rsm.CACHE / "video_cut_1080p30_bbox.npz")["bbox"][F0:F1 + 1]
    K = np.load(rsm.CACHE / "video_cut_1080p30_vggt_camera.npz")["K"][F0:F1 + 1]

    t_dec = time.perf_counter()
    cap = cv2.VideoCapture(str(rsm.VIDEO))
    assert cap.isOpened()
    frames = []
    for i in range(F1 + 1):
        if not cap.grab():
            break
        if i >= F0:
            ok, im = cap.retrieve()
            assert ok
            frames.append(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    cap.release()
    assert len(frames) == T
    decode_s = time.perf_counter() - t_dec
    print(f"[setup] decoded {len(frames)} frames in {decode_s:.1f}s", flush=True)

    base_feat = torch.load(P / "outputs" / "sam_hand_modes" / "feat_full.pt",
                           map_location="cpu", weights_only=True)
    base_body = base_feat[:, :1024]
    base_left = base_feat[:, 1024:2048]
    base_right = base_feat[:, 2048:]

    ext = rsm.build_sam3d_extractor(torch.device("cuda"))
    est = ext.estimator
    model = est.model
    original_step = model.forward_step

    report = {
        "plan": "logs/SPARSE_MEAN_HAND_PLAN.md (cost clause)",
        "frames": T, "decode_seconds": decode_s, "sampling_seed": SEED,
        "note": ("Timings are same-session and include decoding; the mean accumulation is "
                 "inside the timed region. Body tokens are checked against the cached baseline."),
        "schedules": {},
    }

    for name, n in SCHEDULES:
        if n is None:
            hand_set = None                      # every frame
        elif n == 0:
            hand_set = np.array([], dtype=int)   # never
        else:
            hand_set = sampled_indices(T, n, SEED)
        hand_lookup = None if hand_set is None else set(int(x) for x in hand_set.tolist())
        counts = {"body": 0, "hand": 0}

        def counted(batch, decoder_type="body"):
            counts[decoder_type] = counts.get(decoder_type, 0) + 1
            return original_step(batch, decoder_type=decoder_type)

        model.forward_step = counted
        body_tok = torch.zeros(T, 1024, dtype=torch.float32)
        hand_l = torch.zeros(T, 1024, dtype=torch.float32)
        hand_r = torch.zeros(T, 1024, dtype=torch.float32)
        got = np.zeros(T, dtype=bool)
        t0 = time.perf_counter()
        try:
            for i in range(T):
                use_hand = True if hand_lookup is None else (i in hand_lookup)
                ext._body_tokens.clear()
                ext._hand_tokens.clear()
                with rsm.contextlib.redirect_stdout(rsm.Silent()):
                    est.process_one_image(
                        frames[i], bboxes=bbox[i].reshape(1, 4),
                        cam_int=torch.from_numpy(K[i].reshape(1, 3, 3).astype(np.float32)),
                        inference_type="full" if use_hand else "body",
                    )
                body_tok[i] = ext._body_tokens[0][0, 0, :].detach().cpu().float()
                if use_hand:
                    hb = ext._hand_tokens[0]
                    assert hb.shape[0] >= 2, f"merged hand batch expected at {i}"
                    hand_l[i] = hb[0, 0, :].detach().cpu().float()
                    hand_r[i] = hb[1, 0, :].detach().cpu().float()
                    got[i] = True
                if i % 100 == 0 or i == T - 1:
                    el = time.perf_counter() - t0
                    print(f"  [{name}] {i + 1}/{T} {el:.1f}s ({el / (i + 1) * 1000:.0f} ms/frame)",
                          flush=True)
            # mean accumulation is part of the real cost and stays inside the timer
            finite = np.isfinite(hand_l.numpy()).all(axis=1) & np.isfinite(hand_r.numpy()).all(axis=1)
            good = np.flatnonzero(got & finite)
            mean_l = hand_l[good].mean(dim=0, keepdim=True) if good.size else torch.zeros(1, 1024)
            mean_r = hand_r[good].mean(dim=0, keepdim=True) if good.size else torch.zeros(1, 1024)
        finally:
            model.forward_step = original_step
        seconds = time.perf_counter() - t0

        body_diff = float((body_tok - base_body).abs().max())
        entry = {
            "hand_forwards": counts["hand"], "body_forwards": counts["body"],
            "hand_frames_used_for_mean": int(good.size),
            "total_seconds": seconds, "ms_per_frame": seconds / T * 1000,
            "body_token_max_abs_diff_vs_cache": body_diff,
        }
        if good.size:
            dl = float((mean_l - base_left.mean(dim=0, keepdim=True)).abs().max())
            dr = float((mean_r - base_right.mean(dim=0, keepdim=True)).abs().max())
            entry["mean_left_diff_vs_cache_mean"] = dl
            entry["mean_right_diff_vs_cache_mean"] = dr
        report["schedules"][name] = entry
        print(f"[done] {name}: {json.dumps(entry)}", flush=True)

    full_s = report["schedules"]["full"]["total_seconds"]
    for k, v in report["schedules"].items():
        if k == "full":
            continue
        v["speedup_vs_full"] = full_s / v["total_seconds"]
    report["decode_seconds"] = decode_s
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[done] " + str(REPORT), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
