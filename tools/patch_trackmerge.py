# -*- coding: utf-8 -*-
"""修补: ByteTrack 把同一个人切成多段轨迹时, 只取最长的一条会导致覆盖不全。

实测 (video_cut_1080p30, 3794 帧):
  id=1 len=1051 range=[   0,1052] avg_area=267763   <- 同一人的前半段
  id=3 len=2702 range=[1047,3788] avg_area=268480   <- 同一人的后半段
  id=2 len=2    range=[ 469, 470] avg_area=136786   <- 误检

  select_best_track 只按 (length, avg_area) 取一条 -> 选中 id=3 -> 只覆盖 72.3%
  合并 id=1+id=3 后覆盖 3752/3794 = 98.9%

修法: 新增 merge_person_tracks(), 贪婪合并"帧范围重叠或在 GAP 内"且"平均面积接近
      (比值<=1.6)"的轨迹; select_best_track 返回合并后的 TrackInfo。
      面积判据用于排除误检(id=2 面积仅真人的一半)。
"""
import os
import shutil
import sys
from pathlib import Path

TARGET = (Path(__file__).resolve().parent.parent
          / "app" / "flowhmr" / "flowhmr" / "utils" / "runtime_video_processing.py")
BACKUP_SUFFIX = ".orig_trackmerge"
MARKER = "def merge_person_tracks"

MERGE_FN = '''    @staticmethod
    def merge_person_tracks(result: "TrackingResult", min_len: int = 30,
                            gap: int = 45, area_ratio_max: float = 1.6):
        """TRACKFIX: 合并同一人的分段轨迹。

        ByteTrack 在人物短暂丢失后会分配新 track_id, 同一人被切成多条轨迹;
        select_best_track 只取最长的一条会丢失其余片段。
        判据: (a) 帧范围重叠或相距在 gap 内; (b) 平均面积比值 <= area_ratio_max
              (面积差异大的多半是误检, 例如实测 id=2 面积仅真人的一半)。
        """
        if not result.tracks:
            return None
        cands = [t for t in result.tracks if t.length >= min_len] or list(result.tracks)
        if not cands:
            return None
        cands = sorted(cands, key=lambda t: (-t.length, t.track_id))

        seed = cands[0]
        lo, hi = int(seed.frames.min()), int(seed.frames.max())
        frames = list(seed.frames.tolist())
        boxes = list(seed.bboxes.tolist())
        area0 = float(seed.avg_area)

        # 反复吸收与当前区间相邻/重叠且面积接近的轨迹
        progress = True
        while progress:
            progress = False
            for t in cands:
                if t.track_id == seed.track_id:
                    continue
                t_lo, t_hi = int(t.frames.min()), int(t.frames.max())
                if t_hi < lo - gap or t_lo > hi + gap:
                    continue
                ratio = max(float(t.avg_area), area0) / max(min(float(t.avg_area), area0), 1e-6)
                if ratio > area_ratio_max:
                    continue
                frames.extend(t.frames.tolist())
                boxes.extend(t.bboxes.tolist())
                lo = min(lo, t_lo)
                hi = max(hi, t_hi)
                progress = True
            cands = [t for t in cands if t.track_id == seed.track_id or
                     int(t.frames.max()) < lo - gap or int(t.frames.min()) > hi + gap]

        # 同一帧可能被多条轨迹覆盖 -> 去重(保留后出现的)
        frame_to_box = {}
        for f, b in zip(frames, boxes):
            frame_to_box[int(f)] = b
        fr = np.asarray(sorted(frame_to_box.keys()), dtype=np.int64)
        bx = np.asarray([frame_to_box[int(f)] for f in fr], dtype=np.float32)
        areas = (bx[:, 2] - bx[:, 0]) * (bx[:, 3] - bx[:, 1])
        return TrackInfo(track_id=seed.track_id, frames=fr, bboxes=bx,
                         length=int(fr.size), avg_area=float(np.mean(areas)))

'''


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else TARGET
    if not os.path.isfile(path):
        print(f"FAIL 找不到: {path}")
        return 1
    src = open(path, encoding="utf-8").read()
    if MARKER in src:
        print("SKIP 已打过补丁(幂等)")
        return 0

    bak = path + BACKUP_SUFFIX
    if not os.path.isfile(bak):
        shutil.copy2(path, bak)
        print(f"OK 备份 -> {bak}")

    # 1. 在 select_best_track 前插入 merge_person_tracks
    anchor = "    @staticmethod\n    def select_best_track(result: TrackingResult) -> int:\n"
    if anchor not in src:
        print("FAIL 找不到 select_best_track 锚点")
        return 1
    src = src.replace(anchor, MERGE_FN + anchor, 1)
    print("OK 已插入 merge_person_tracks")

    # 2. select_best_track 内部改用合并结果
    old_body = (
        '        if not result.tracks:\n'
        '            return -1\n'
        '        scored = [(i, t.length, t.avg_area) for i, t in enumerate(result.tracks)]\n'
        '        scored.sort(key=lambda x: (x[1], x[2]), reverse=True)\n'
        '        return scored[0][0]\n'
    )
    new_body = (
        '        if not result.tracks:\n'
        '            return -1\n'
        '        merged = VideoProcessor.merge_person_tracks(result)  # TRACKFIX: 合并分段轨迹\n'
        '        if merged is not None:\n'
        '            for i, t in enumerate(result.tracks):\n'
        '                if t.track_id == merged.track_id:\n'
        '                    result.tracks[i] = merged\n'
        '                    print(f">>> TRACKFIX merged: track {merged.track_id} -> "\n'
        '                          f"{merged.length} frames, [{int(merged.frames.min())},{int(merged.frames.max())}], "\n'
        '                          f"avg_area={merged.avg_area:.0f}")\n'
        '                    return i\n'
        '        scored = [(i, t.length, t.avg_area) for i, t in enumerate(result.tracks)]\n'
        '        scored.sort(key=lambda x: (x[1], x[2]), reverse=True)\n'
        '        return scored[0][0]\n'
    )
    if old_body not in src:
        print("FAIL 找不到 select_best_track 函数体")
        return 1
    src = src.replace(old_body, new_body, 1)
    print("OK select_best_track 已改用合并结果")

    open(path, "w", encoding="utf-8").write(src)

    # 3. 自检
    chk = open(path, encoding="utf-8").read()
    checks = [
        ("merge_person_tracks 已定义", "def merge_person_tracks(" in chk),
        ("select_best_track 调用合并", "merge_person_tracks(result)" in chk),
        ("TrackInfo/VideoProcessor 可用", "class TrackInfo" in chk and "class VideoProcessor" in chk),
        ("interpolate_track 未动", "def interpolate_track(" in chk),
        ("compute_bboxes 未动", "def compute_bboxes(" in chk),
        ("detect_and_track 未动", "def detect_and_track(" in chk),
    ]
    print("\n--- 语义自检 ---")
    ok_all = True
    for name, ok in checks:
        print(f"  {'OK  ' if ok else 'FAIL'} {name}")
        ok_all = ok_all and ok
    import py_compile
    try:
        py_compile.compile(path, doraise=True)
        print("  OK   语法编译通过")
    except Exception as e:
        print(f"  FAIL 语法错误: {e}")
        ok_all = False
    print("\nPATCH_OK" if ok_all else "\nPATCH_FAILED")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
