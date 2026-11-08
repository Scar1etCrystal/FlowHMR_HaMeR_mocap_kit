"""Measure generation cost/VRAM for cfg_scale 1 vs 3 at several sequence lengths.

Uses the already extracted SAM tokens of finished runs, so only the diffusion
generation is timed. Engine is built without postprocess (CPU stage, irrelevant).
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

PACK = Path(r'F:\flowhmr_pack')
APP = PACK / 'app' / 'flowhmr'
import os

# Same attention configuration the production runner exports before importing.
os.environ['PYTHONUTF8'] = '1'
os.environ.setdefault('MHR_USE_CUDA_GRAPH', '0')
os.environ.setdefault('USE_COMPILE', '0')
os.environ['FLOWHMR_ATTENTION_BACKEND'] = 'query_batched'
os.environ['FLOWHMR_ATTENTION_QUERY_CHUNK_SIZE'] = '256'
sys.path.insert(0, str(APP))
os.chdir(APP)

from flowhmr.pipeline.v2m_inference import (  # noqa: E402
    V2MInferenceEngine, build_feature_dict)

CKPT = APP / 'checkpoints' / 'flowhmr_latest' / 'flowhmr_latest.ckpt'
CFG = APP / 'configs' / 'base' / 'model_basic05B.yml'

FEATURES = {
    # One representative long take is enough for the cfg comparison.
    6929: PACK / 'outputs/video_to_fbx/ribbon_1080p30_135846b63b/official/ribbon_1080p30/ribbon_1080p30_sam3d_feat_static.pt',
}

engine = V2MInferenceEngine.build(ckpt=str(CKPT), model_cfg_path=str(CFG), device='cuda',
                                  validation_steps=None, postprocess=False)
rows = []
for label, path in FEATURES.items():
    if not path.is_file():
        print('MISSING', path)
        continue
    tokens = torch.load(path, map_location='cpu', weights_only=True)
    if tokens.dim() == 3:
        tokens = tokens[0]
    length = int(tokens.shape[0])
    camera = torch.eye(4, dtype=torch.float32).unsqueeze(0).repeat(length, 1, 1)
    feature = build_feature_dict(tokens.to(engine.device), camera, engine.device)
    for cfg in (1.0, 3.0):
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated() / 2 ** 30
        import time
        started = time.perf_counter()
        with torch.no_grad():
            engine.pipeline.generate(feature=feature, seeds=[0], length=length, cfg_scale=cfg)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        peak = torch.cuda.max_memory_allocated() / 2 ** 30
        rows.append({'frames': length, 'cfg_scale': cfg, 'generate_s': round(elapsed, 2),
                     'peak_alloc_GiB': round(peak, 2), 'baseline_GiB': round(base, 2)})
        print(f'RESULT frames={length:>5} cfg={cfg:.0f} gen={elapsed:7.2f}s '
              f'peak_alloc={peak:5.2f} GiB (baseline {base:.2f})', flush=True)
    del tokens
    torch.cuda.empty_cache()

out = PACK / 'logs' / 'cfg_scale_generate_bench.json'
out.write_text(json.dumps(rows, indent=2), encoding='utf-8')
print('WROTE', out)
