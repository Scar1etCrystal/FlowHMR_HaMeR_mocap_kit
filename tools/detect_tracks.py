"""Detect + track one video and plan shot segments (no network inference).

Usage:
    python tools/detect_tracks.py --video <mp4> --out <job/tracks.npz> [--plan <json>]

The tracker pass is what the official pipeline would run anyway; caching it lets
the runner cut the video into shots without paying for detection twice.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PACK = Path(__file__).resolve().parents[1]
APP = PACK / 'app' / 'flowhmr'


def configure() -> None:
    os.environ['PYTHONUTF8'] = '1'
    os.environ.setdefault('PYTHONIOENCODING', 'utf-8')
    os.environ.setdefault('MHR_USE_CUDA_GRAPH', '0')
    os.environ.setdefault('USE_COMPILE', '0')
    os.environ.setdefault('PARALLEL_DECODERS', '0')
    for path in (str(APP), str(PACK), str(Path(__file__).resolve().parent)):
        if path not in sys.path:
            sys.path.insert(0, path)
    os.chdir(APP)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--video', required=True)
    ap.add_argument('--out', required=True, help='tracks npz path')
    ap.add_argument('--plan', default='', help='optional plan json path')
    ap.add_argument('--min-frames', type=int, default=24)
    ap.add_argument('--min-share', type=float, default=0.02)
    ap.add_argument('--gap-frames', type=int, default=24)
    args = ap.parse_args()

    configure()
    video = Path(args.video)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    import torch

    import multishot
    from flowhmr.utils.runtime_video_processing import VideoProcessor, build_human_detector

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'  detector device: {device}', flush=True)
    detector = build_human_detector(device)
    proc = VideoProcessor(detector)
    result = proc.detect_and_track(str(video))
    tracks = [{'id': t.track_id, 'frames': t.frames, 'bboxes': t.bboxes} for t in result.tracks]
    multishot.save_tracks(out, tracks, total=result.total_frames,
                          width=result.width, height=result.height)

    segments = multishot.plan_segments(
        tracks, result.total_frames, result.width, result.height,
        min_frames=args.min_frames, min_share=args.min_share, gap_frames=args.gap_frames)
    description = multishot.describe_plan(tracks, result.total_frames, result.width,
                                          result.height, segments)
    print(json.dumps(description, ensure_ascii=False, indent=2), flush=True)
    if args.plan:
        Path(args.plan).write_text(json.dumps(description, ensure_ascii=False, indent=2),
                                   encoding='utf-8')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

