# -*- coding: utf-8 -*-
"""探测「生成阶段显存 vs 帧数」，用于回答 12 GB 卡一次能跑多少帧。

做法：只 build 一次引擎，用已缓存的 SAM 特征截断成不同长度，逐次调用 run_generation，
每次 reset_peak_memory_stats 后记录峰值 allocated。

关键：测之前把 SAM / YOLOX 释放掉，模拟真实场景（阶段间卸载后的生成阶段）。
"""
from __future__ import annotations

import gc
import json
import os
import sys
import time

import torch

REPO = r"F:\flowhmr_pack\app\flowhmr"
os.chdir(REPO)
sys.path.insert(0, REPO)

OUT = r"F:\flowhmr_pack\logs\vram_vs_frames.json"
FEAT = os.path.join(REPO, "output", "chunk_full", "chunk_full_sam3d_feat.pt")
TMP = os.path.join(REPO, "output", "_vramprobe")

LENGTHS = [60, 120, 180, 240, 300]


def main() -> int:
    from flowhmr.pipeline.v2m_inference import V2MInferenceEngine

    t0 = time.time()
    engine = V2MInferenceEngine.build(
        ckpt=os.path.join(REPO, "checkpoints", "flowhmr_latest", "flowhmr_latest.ckpt"),
        model_cfg_path=os.path.join(REPO, "configs", "base", "model_basic05B.yml"),
        device="cuda", validation_steps=None, postprocess=False,
    )
    print(f"[probe] engine built in {time.time()-t0:.1f}s", flush=True)

    # 模拟阶段间卸载后的状态：放掉 SAM3D + YOLOX
    engine.sam_extractor = None
    try:
        engine.proc.detector = None
    except Exception:
        pass
    gc.collect()
    torch.cuda.empty_cache()
    base = torch.cuda.memory_allocated() / 1024**3
    print(f"[probe] baseline after unload: {base:.3f} GiB allocated", flush=True)

    feats = torch.load(FEAT, map_location="cpu", weights_only=True)
    feats = feats.to(torch.float32).contiguous()
    print(f"[probe] cached features: {tuple(feats.shape)}", flush=True)

    os.makedirs(TMP, exist_ok=True)
    rows = []
    for T in LENGTHS:
        if T > feats.shape[0]:
            print(f"[probe] skip T={T} (features only {feats.shape[0]})")
            continue
        tok = feats[:T].clone()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        b0 = torch.cuda.memory_allocated() / 1024**3
        t1 = time.time()
        engine.run_generation(tok, TMP, f"probe{T}", seed=0, cfg_scale=1.0, export_mesh=False)
        dt = time.time() - t1
        peak_a = torch.cuda.max_memory_allocated() / 1024**3
        peak_r = torch.cuda.max_memory_reserved() / 1024**3
        rows.append({"T": T, "base_gib": round(b0, 3), "peak_alloc_gib": round(peak_a, 4),
                     "peak_reserved_gib": round(peak_r, 4), "gen_s": round(dt, 1)})
        print(f"[probe] T={T:5d}  base={b0:.3f}  peak_alloc={peak_a:.3f} GiB  "
              f"peak_reserved={peak_r:.3f} GiB  {dt:.1f}s", flush=True)
        del tok
        gc.collect()
        torch.cuda.empty_cache()

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump({"baseline_gib": base, "rows": rows}, f, ensure_ascii=False, indent=2)

    # 拟合 增值 = a * T^b
    import numpy as np
    Tl = np.array([r["T"] for r in rows], float)
    dA = np.array([r["peak_alloc_gib"] - r["base_gib"] for r in rows], float)
    m = dA > 0
    if m.sum() >= 2:
        b, loga = np.polyfit(np.log(Tl[m]), np.log(dA[m]), 1)
        a = np.exp(loga)
        print(f"\n[probe] 拟合: 增值 = {a:.3e} * T^{b:.3f}")
        with open(OUT, "w", encoding="utf-8") as f:
            json.dump({"baseline_gib": base, "rows": rows,
                       "fit": {"a": float(a), "b": float(b)}}, f, ensure_ascii=False, indent=2)
    print(f"[probe] -> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
