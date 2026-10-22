"""Measure how much VRAM VGGT needs as the number of views grows.

Runs the real camera extraction on the real clip with a few intervals, recording
peak allocated VRAM after each, and stops at the first out-of-memory.

    envs\\python.exe tools/measure_vggt_vram.py <job_dir> <frames>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

PACK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACK / 'app' / 'flowhmr'))

from flowhmr.utils.runtime_vggt_camera import extract_vggt_camera, release_vggt_model  # noqa: E402


def sample_count(frames: int, interval: int) -> int:
    n = len(range(0, frames, interval))
    if (frames - 1) % interval != 0:
        n += 1
    return n


def main():
    job = Path(sys.argv[1])
    frames = int(sys.argv[2]) if len(sys.argv) > 2 else 2764
    video = str(job / 'working_30fps.mp4')
    print(f'video: {video}', flush=True)
    print(f'frames: {frames}', flush=True)
    total = torch.cuda.get_device_properties(0).total_memory / 2**30
    print(f'gpu total: {total:.2f} GiB', flush=True)

    rows = []
    intervals = [64, 48, 32, 24, 16, 12, 10, 8, 6, 5, 4, 3, 2]
    for iv in intervals:
        n = sample_count(frames, iv)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated() / 2**30
        try:
            extract_vggt_camera(video, output_npz=None, frame_range=(0, frames - 1),
                                expected_len=frames, device='cuda', frame_interval=iv)
            peak = torch.cuda.max_memory_allocated() / 2**30
            reserved = torch.cuda.max_memory_reserved() / 2**30
            rows.append({'interval': iv, 'samples': n, 'peak_GiB': round(peak, 3),
                         'reserved_GiB': round(reserved, 3), 'ok': True})
            print(f'MEASURE iv={iv:<3} samples={n:<5} peak={peak:6.2f} GiB  reserved={reserved:6.2f}'
                  f'  (base {base:.2f})', flush=True)
        except torch.OutOfMemoryError as exc:
            rows.append({'interval': iv, 'samples': n, 'ok': False,
                         'error': str(exc).splitlines()[0][:120]})
            print(f'MEASURE iv={iv:<3} samples={n:<5} OUT OF MEMORY', flush=True)
            torch.cuda.empty_cache()
            break
    out = PACK / 'logs' / 'vggt_vram_curve.json'
    out.write_text(json.dumps({'frames': frames, 'gpu_total_GiB': round(total, 3), 'rows': rows},
                              indent=2), encoding='utf-8')
    print(f'\nwritten {out}', flush=True)


if __name__ == '__main__':
    main()
