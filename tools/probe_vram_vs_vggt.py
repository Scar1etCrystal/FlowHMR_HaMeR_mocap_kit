# -*- coding: utf-8 -*-
"""探测 VGGT 相机阶段的显存 vs 采样帧数。

VGGT 把采样到的帧**联合**处理，显存随采样帧数增长；
采样帧数 = 视频帧数 / --vggt_interval。这是 12 GB 卡真正的约束。
"""
from __future__ import annotations

import json
import os
import sys

import torch

REPO = r"F:\flowhmr_pack\app\flowhmr"
os.chdir(REPO)
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "third_party", "vggt-omega"))

VIDEO = os.path.join(REPO, "output", "chunk_full", "chunk_full_30fps.mp4")   # 300 帧
OUT = r"F:\flowhmr_pack\logs\vram_vs_vggt_samples.json"
INTERVALS = [10, 5, 2, 1]          # -> 采样 30 / 60 / 150 / 300 帧


def main() -> int:
    from flowhmr.utils.runtime_vggt_camera import extract_vggt_camera, release_vggt_model

    rows = []
    for iv in INTERVALS:
        n_samples = 300 // iv
        tmp = os.path.join(REPO, "output", "_vggtprobe", f"iv{iv}.npz")
        os.makedirs(os.path.dirname(tmp), exist_ok=True)
        if os.path.exists(tmp):
            os.remove(tmp)
        release_vggt_model()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        b0 = torch.cuda.memory_allocated() / 1024**3
        try:
            extract_vggt_camera(
                VIDEO, output_npz=tmp, frame_range=(0, 299), expected_len=300,
                device="cuda", frame_interval=iv,
            )
            peak = torch.cuda.max_memory_allocated() / 1024**3
            res = torch.cuda.max_memory_reserved() / 1024**3
            rows.append({"interval": iv, "samples": n_samples,
                         "base_gib": round(b0, 3), "peak_alloc_gib": round(peak, 4),
                         "peak_reserved_gib": round(res, 4), "ok": True})
            print(f"[vggt] interval={iv:2d} samples={n_samples:3d}  base={b0:.3f}  "
                  f"peak_alloc={peak:.3f} GiB  reserved={res:.3f} GiB", flush=True)
        except RuntimeError as e:
            rows.append({"interval": iv, "samples": n_samples, "ok": False,
                         "err": str(e)[:200]})
            print(f"[vggt] interval={iv:2d} samples={n_samples:3d}  ** OOM/失败: {str(e)[:120]}", flush=True)
            torch.cuda.empty_cache()

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    import numpy as np
    ok = [r for r in rows if r.get("ok")]
    if len(ok) >= 2:
        S = np.array([r["samples"] for r in ok], float)
        D = np.array([r["peak_alloc_gib"] - r["base_gib"] for r in ok], float)
        m = D > 0
        if m.sum() >= 2:
            b, loga = np.polyfit(np.log(S[m]), np.log(D[m]), 1)
            print(f"\n[vggt] 拟合: 增值 = {np.exp(loga):.3e} * samples^{b:.3f}")
    print(f"[vggt] -> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
