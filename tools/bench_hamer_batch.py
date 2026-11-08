"""Benchmark HaMeR batch size on an existing job's working clip.

Usage: python tools/bench_hamer_batch.py <job_with_working_video> [batch,total_batches]
Runs the real hamer_pose_sampled.py stage with different --batch_size values and
reports wall time and peak GPU memory from hands/run_info.json.
"""
import json
import re
import subprocess
import sys
import time
from pathlib import Path

PACK = Path(__file__).resolve().parents[1]
PY = PACK / 'envs' / 'python.exe'


def main() -> int:
    job = Path(sys.argv[1])
    batches = [int(value) for value in sys.argv[2].split(',')] if len(sys.argv) > 2 else [16, 32, 48, 64]
    video = job / 'working_30fps.mp4'
    track = job / 'main_track.npz'
    if not video.is_file() or not track.is_file():
        raise SystemExit(f'missing inputs in {job}')
    rows = []
    for batch in batches:
        out_dir = job / f'hands_batch{batch}'
        cmd = [str(PY), '-u', str(PACK / 'tools' / 'hamer_pose_sampled.py'),
               '--video', str(video), '--main_track', str(track),
               '--out_dir', str(out_dir), '--detect_interval', '2',
               '--batch_size', str(batch), '--window', '11', '--max_interp_gap', '15']
        started = time.perf_counter()
        proc = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace')
        elapsed = time.perf_counter() - started
        info = {}
        info_path = out_dir / 'run_info.json'
        if info_path.is_file():
            info = json.loads(info_path.read_text(encoding='utf-8'))
        rows.append({'batch': batch, 'elapsed_s': round(elapsed, 1),
                     'peak_GiB': info.get('peak_alloc_GiB'),
                     'frames': info.get('frames'), 'ok': proc.returncode == 0})
        print(f'  batch {batch:>3}: {elapsed:7.1f}s  peak {info.get("peak_alloc_GiB")} GiB  '
              f'rc={proc.returncode}', flush=True)
        if proc.returncode != 0:
            tail = '\n'.join((proc.stderr or '').strip().splitlines()[-5:])
            print('   ', tail.replace('\n', '\n    '), flush=True)
    report = job / 'hamer_batch_bench.json'
    report.write_text(json.dumps(rows, indent=2), encoding='utf-8')
    print('WROTE', report)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
