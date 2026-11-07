# -*- coding: utf-8 -*-
"""分块处理测试：切分素材并生成各组片段。

设计（300 帧 = 10 s @30fps 为基准）:
  单次    full        [0,299]                        ← 对照组
  截断    truncA      [0,149]     truncB [150,299]   ← 无重叠，直接拼接
  重叠    overA       [0,194]     overB  [105,299]   ← 90 帧(=3 s) 重叠

输出到 inputs/chunktest/，供 run_flowhmr.bat 一次跑完。
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

PACK = Path(__file__).resolve().parent.parent
SRC = PACK / "inputs" / "source.mp4"          # 568 帧 720x1280 30.17fps
OUT = PACK / "inputs" / "chunktest"

N_FULL = 300
OVERLAP = 90          # 3 s @ 30fps
MID = N_FULL // 2     # 150
SEG_A = MID + OVERLAP // 2        # 195
SEG_B_START = MID - OVERLAP // 2  # 105


def ffmpeg() -> str:
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def cut(src: str, dst: str, start: int, count: int, fps=30) -> None:
    cmd = [ffmpeg(), "-y", "-v", "error",
           "-i", src,
           "-vf", f"fps={fps},select='between(n\\,{start}\\,{start + count - 1})',setpts=N/{fps}/TB",
           "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
           "-pix_fmt", "yuv420p", dst]
    subprocess.run(cmd, check=True)


def main() -> int:
    if not SRC.is_file():
        print(f"FAIL 找不到素材: {SRC}")
        return 1
    OUT.mkdir(parents=True, exist_ok=True)

    # 先转出统一的 30fps 全片，再切（避免 select 在 VFR 上错位）
    full = str(OUT / "chunk_full.mp4")
    cut(str(SRC), full, 0, N_FULL)
    print(f"OK full      [0,{N_FULL - 1}]           -> {full}")

    jobs = [
        ("truncA", 0, MID),
        ("truncB", MID, N_FULL - MID),
        ("overA", 0, SEG_A),
        ("overB", SEG_B_START, N_FULL - SEG_B_START),
    ]
    for name, start, count in jobs:
        dst = str(OUT / f"chunk_{name}.mp4")
        cut(full, dst, start, count)
        print(f"OK {name:8s} [{start},{start + count - 1}]  ({count} 帧) -> {dst}")

    print(f"\n重叠: {OVERLAP} 帧 = {OVERLAP / 30:.1f} s")
    print(f"截断: truncA[0,{MID - 1}] + truncB[{MID},{N_FULL - 1}]  (无缝无重叠)")
    print(f"重叠: overA[0,{SEG_A - 1}] + overB[{SEG_B_START},{N_FULL - 1}]  "
          f"(重叠 [{SEG_B_START},{SEG_A - 1}] = {SEG_A - SEG_B_START} 帧)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
