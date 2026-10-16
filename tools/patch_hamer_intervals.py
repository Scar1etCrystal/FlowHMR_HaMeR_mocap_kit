# -*- coding: utf-8 -*-
"""把 hamer_locked.py 的"隔 2 帧检测+插值"对照组参数化为"隔 N 帧"。

原实现只写死 i2（f % 2 == 0）；这里改成 --intervals 2,3,... 一次跑出多组对照，
输出 <stem>_hamer_data_i{N}.pkl，与主结果 <stem>_hamer_data.pkl（逐帧）比较。

幂等；原文件备份为 .orig_pack。

用法: envs\\python.exe tools\\patch_hamer_intervals.py
"""
from __future__ import annotations

import py_compile
import shutil
import sys
from pathlib import Path

PACK = Path(__file__).resolve().parent.parent
TARGET = PACK / "app" / "hamer" / "hamer_locked.py"
MARKER = "cmp_tasks"

# 1) argparse: 增加 --intervals
OLD_ARG = '''    p.add_argument("--max_frames", type=int, default=0, help="调试：只跑前 N 帧")
    return p.parse_args()'''
NEW_ARG = '''    p.add_argument("--max_frames", type=int, default=0, help="调试：只跑前 N 帧")
    p.add_argument("--intervals", default="2",
                   help="对照组：隔 N 帧检测+中间插值，逗号分隔（如 2,3）；主结果始终是逐帧")
    return p.parse_args()'''

# 2) 对照组任务构建
OLD_TASKS = '''    # 对照：锁主角、但照原版隔 2 帧检测（只用偶数帧的检测框，其余插值）→ 拆开"锁主角"和"每帧检测"各自的作用
    tasks_i2 = []
    for h in (0, 1):
        fr = [f for f in sorted(det_box[h]) if f % 2 == 0]
        if not fr:
            continue
        fn = interp1d(fr, [det_box[h][f] for f in fr], axis=0, kind="linear", fill_value="extrapolate")
        for f in range(min(fr), max(fr) + 1):
            same = f % 2 == 0 and f in det_box[h]                         # 与主结果同一个框 → 直接复用
            tasks_i2.append({"frame_idx": f, "bbox": det_box[h][f] if same else fn(f).astype(np.float32),
                             "is_right": h, "results": {}, "reuse": same})'''
NEW_TASKS = '''    # 对照：锁主角、但隔 N 帧检测（只用 f % N == 0 的检测框，其余线性插值）
    # → 拆开"锁主角"和"每帧检测"各自的作用
    intervals = [int(x) for x in str(a.intervals).split(",") if x.strip()]
    cmp_tasks = {}
    for _N in intervals:
        if _N <= 1:
            continue
        _ts = []
        for h in (0, 1):
            fr = [f for f in sorted(det_box[h]) if f % _N == 0]
            if not fr:
                continue
            fn = interp1d(fr, [det_box[h][f] for f in fr], axis=0, kind="linear", fill_value="extrapolate")
            for f in range(min(fr), max(fr) + 1):
                same = f % _N == 0 and f in det_box[h]                    # 与主结果同一个框 → 直接复用
                _ts.append({"frame_idx": f, "bbox": det_box[h][f] if same else fn(f).astype(np.float32),
                            "is_right": h, "results": {}, "reuse": same})
        cmp_tasks[_N] = _ts
        _nd = sum(1 for t in _ts if t["reuse"])
        print(f"[cmp] interval={_N}: {len(_ts)} tasks, {_nd} reused from per-frame run, "
              f"{len(_ts) - _nd} extra HaMeR calls", flush=True)'''

# 3) 分帧索引
OLD_IDX = '''    i2_by_frame = {}
    for t in tasks_i2:
        if not t["reuse"]:
            i2_by_frame.setdefault(t["frame_idx"], []).append(t)'''
NEW_IDX = '''    cmp_by_frame = {}
    for _N, _ts in cmp_tasks.items():
        _d = {}
        for t in _ts:
            if not t["reuse"]:
                _d.setdefault(t["frame_idx"], []).append(t)
        cmp_by_frame[_N] = _d'''

# 4) 入队
OLD_ADD = '''        for t in i2_by_frame.get(f, []):
            hb.add(("i2", f, t["is_right"]), frame, t["bbox"], t["is_right"])'''
NEW_ADD = '''        for _N, _d in cmp_by_frame.items():
            for t in _d.get(f, []):
                hb.add(("cmp", _N, f, t["is_right"]), frame, t["bbox"], t["is_right"])'''

# 5) 导出
OLD_EXPORT = '''    for t in tasks_i2:
        r = hb.out[("m", t["frame_idx"], t["is_right"], 0)] if t["reuse"] else hb.out[("i2", t["frame_idx"], t["is_right"])]
        t["results"] = {k: r[k] for k in ("pose", "orient", "betas", "cam_t")}
    with open(out_dir / f"{stem}_hamer_data_i2.pkl", "wb") as fh:
        pickle.dump(export_like_original(tasks_i2, num_frames, a.window), fh, protocol=4)'''
NEW_EXPORT = '''    for _N, _ts in cmp_tasks.items():
        for t in _ts:
            _key = (("m", t["frame_idx"], t["is_right"], 0) if t["reuse"]
                    else ("cmp", _N, t["frame_idx"], t["is_right"]))
            r = hb.out[_key]
            t["results"] = {k: r[k] for k in ("pose", "orient", "betas", "cam_t")}
        _out = out_dir / f"{stem}_hamer_data_i{_N}.pkl"
        with open(_out, "wb") as fh:
            pickle.dump(export_like_original(_ts, num_frames, a.window), fh, protocol=4)
        print(f"[3/3] exported {_out.name}  ({len(_ts)} tasks)", flush=True)'''

SUBS = [
    ("argparse", OLD_ARG, NEW_ARG),
    ("cmp-tasks", OLD_TASKS, NEW_TASKS),
    ("cmp-index", OLD_IDX, NEW_IDX),
    ("cmp-add", OLD_ADD, NEW_ADD),
    ("cmp-export", OLD_EXPORT, NEW_EXPORT),
]


def main() -> int:
    if not TARGET.exists():
        print(f"FAIL 找不到 {TARGET}")
        return 1
    src = TARGET.read_text(encoding="utf-8")
    if MARKER in src:
        print("SKIP 已打过（幂等）")
        return 0

    todo = []
    for name, old, new in SUBS:
        cnt = src.count(old)
        if cnt != 1:
            print(f"FAIL [{name}] 锚点出现 {cnt} 次（需 1 次）")
            return 1
        todo.append((name, old, new))

    bak = TARGET.with_suffix(".py.orig_pack")
    if not bak.exists():
        shutil.copy2(TARGET, bak)
        print(f"OK 备份 -> {bak.name}")

    for name, old, new in todo:
        src = src.replace(old, new, 1)
        print(f"OK [{name}] 已替换")
    TARGET.write_text(src, encoding="utf-8")

    try:
        py_compile.compile(str(TARGET), doraise=True)
        print("OK 语法编译通过")
    except py_compile.PyCompileError as e:
        print(f"FAIL 语法错误, 回滚: {e}")
        shutil.copy2(bak, TARGET)
        return 1

    chk = TARGET.read_text(encoding="utf-8")
    for name, ok in (("cmp_tasks 已定义", "cmp_tasks = {}" in chk),
                     ("--intervals 已加", '"--intervals"' in chk),
                     ("i{N} 导出", "_hamer_data_i{_N}.pkl" in chk),
                     ("主结果导出未动", "_hamer_data.pkl" in chk)):
        print(f"  {'OK  ' if ok else 'FAIL'} {name}")
    print("\nPATCH_OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
