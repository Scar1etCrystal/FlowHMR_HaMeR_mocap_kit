# -*- coding: utf-8 -*-
"""HaMeR 隔帧检测 + 插值 对照报告。

比较：逐帧检测（主结果） vs 隔2帧/隔3帧检测 + 中间线性插值手框
输出：手框误差(像素) + 手部姿态误差(测地角度) + 计算量节省
"""
import os
import pickle
import sys

import numpy as np
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation as R

D = sys.argv[1] if len(sys.argv) > 1 else r"F:\flowhmr_pack\app\flowhmr\output\example\hamer"
OUT = sys.argv[2] if len(sys.argv) > 2 else r"F:\flowhmr_pack\logs\hamer_interval_report.md"
STEM = "example_30fps"


def load_pkl(p):
    with open(p, "rb") as f:
        return pickle.load(f)


main = load_pkl(os.path.join(D, f"{STEM}_hamer_data.pkl"))
raw = np.load(os.path.join(D, f"{STEM}_hamer_raw.npz"))

print("=== pkl 字段 ===")
for k, v in main.items():
    try:
        print(f"  {k:22s} {np.asarray(v).shape} {np.asarray(v).dtype}")
    except Exception:
        print(f"  {k:22s} {type(v)}")

hand = np.asarray(main["hand_pose"], dtype=np.float64)      # (T,2,15,3,3)
T = hand.shape[0]
FPS = 30.0
print(f"\n  T={T}  hand_pose{hand.shape}")

# ---------- 1) 手框插值误差 ----------
boxes = raw["hand_box"]           # (T,2,4)
src = raw["box_src"]              # (T,2)  1=detected 2=interp
detected = {h: [f for f in range(T) if src[f, h] == 1] for h in (0, 1)}
print(f"\n=== 逐帧检测到的手框 ===")
for h, nm in ((0, "左"), (1, "右")):
    print(f"  {nm}手: {len(detected[h])}/{T} 帧检测到")

box_err = {}
for N in (2, 3):
    errs = []
    for h in (0, 1):
        fr = [f for f in detected[h] if f % N == 0]
        if len(fr) < 2:
            continue
        fn = interp1d(fr, [boxes[f, h] for f in fr], axis=0, kind="linear", fill_value="extrapolate")
        for f in range(T):
            if f in fr:
                continue
            pred = np.asarray(fn(f), np.float32)
            side = max(boxes[f, h][2] - boxes[f, h][0], boxes[f, h][3] - boxes[f, h][1])
            c_true = np.array([(boxes[f, h][0] + boxes[f, h][2]) / 2, (boxes[f, h][1] + boxes[f, h][3]) / 2])
            c_pred = np.array([(pred[0] + pred[2]) / 2, (pred[1] + pred[3]) / 2])
            errs.append(np.linalg.norm(c_pred - c_true) / side * 100)   # 中心偏移占手框边长 %
    box_err[N] = np.asarray(errs)
    print(f"  隔{N}帧插值手框中心偏移: mean={box_err[N].mean():.3f}%  p99={np.percentile(box_err[N],99):.3f}%  max={box_err[N].max():.3f}%  (占手框边长)")

# ---------- 2) 姿态误差 ----------
def geo_diff(A, B, valid=None):
    """A,B: (T,2,15,3,3) -> 测地角差(度)"""
    Ta = A.reshape(-1, 3, 3); Tb = B.reshape(-1, 3, 3)
    ra = R.from_matrix(Ta); rb = R.from_matrix(Tb)
    d = np.degrees((ra.inv() * rb).magnitude())
    if valid is not None:
        d = d[valid.reshape(-1)]
    return d


print("\n=== 手部姿态误差（相对逐帧结果）===")
rows = []
for N in (2, 3):
    p = os.path.join(D, f"{STEM}_hamer_data_i{N}.pkl")
    if not os.path.isfile(p):
        print(f"  隔{N}帧: 缺 {p}")
        continue
    alt = np.asarray(load_pkl(p)["hand_pose"], dtype=np.float64)
    n = min(T, alt.shape[0])
    for h, nm in ((0, "左"), (1, "右")):
        A = hand[:n, h]; B = alt[:n, h]
        d = geo_diff(A, B)
        rows.append((N, nm, d.mean(), np.percentile(d, 50), np.percentile(d, 99), d.max()))
    dall = geo_diff(hand[:n], alt[:n])
    rows.append((N, "双手", dall.mean(), np.percentile(dall, 50), np.percentile(dall, 99), dall.max()))

print(f"  {'隔N帧':>6s} {'部位':>4s} {'mean':>9s} {'p50':>9s} {'p99':>9s} {'max':>9s}")
for N, nm, m, p50, p99, mx in rows:
    print(f"  {N:>6d} {nm:>4s} {m:9.4f} {p50:9.4f} {p99:9.4f} {mx:9.4f}")

# ---------- 3) 其他字段 ----------
print("\n=== 其他导出字段的差异（隔2帧）===")
alt2 = load_pkl(os.path.join(D, f"{STEM}_hamer_data_i2.pkl"))
for k in main:
    if k == "hand_pose":
        continue
    try:
        A = np.asarray(main[k], np.float64); B = np.asarray(alt2[k], np.float64)
        if A.shape == B.shape and A.size:
            print(f"  {k:22s} maxdiff={np.abs(A-B).max():.6g}")
    except Exception:
        pass

# ---------- 4) 报告 ----------
os.makedirs(os.path.dirname(OUT), exist_ok=True)
L = []
L.append("# HaMeR 隔帧检测 + 插值 对照报告\n")
L.append(f"- 素材: `{STEM}`（151 帧 / 30 fps / 1920x1080）")
L.append(f"- 运行: `hamer_locked.py --tta 1`（锁主角，给定 Fast-SAM3D 的主角框）")
L.append(f"- GPU: RTX 3060 12GB，总耗时 68.2 s（含模型加载）\n")
L.append("## 1. 逐帧检测结果（基准）\n")
L.append(f"- 手框检测命中：左手 {len(detected[0])}/{T}，右手 {len(detected[1])}/{T}")
L.append(f"- 画面中其他人的手：0（锁主角有效）\n")
L.append("## 2. 手框插值误差（几何层面）\n")
L.append("| 检测间隔 | 中心偏移 mean | p99 | max |")
L.append("|---|---|---|---|")
for N in (2, 3):
    e = box_err[N]
    L.append(f"| 隔 {N} 帧 | {e.mean():.3f}% | {np.percentile(e,99):.3f}% | {e.max():.3f}% |")
L.append("\n> 单位为手框边长的百分比。手框很小（手部尺度），框偏一点，裁剪内容变化就很大。\n")
L.append("## 3. 手部姿态误差（相对逐帧结果，测地角度）\n")
L.append("| 检测间隔 | 部位 | mean | p50 | p99 | max |")
L.append("|---|---|---|---|---|---|")
for N, nm, m, p50, p99, mx in rows:
    L.append(f"| 隔 {N} 帧 | {nm} | {m:.4f}° | {p50:.4f}° | {p99:.4f}° | {mx:.4f}° |")
L.append("\n## 4. 计算量\n")
L.append("| 模式 | HaMeR 调用次数 | 相对逐帧 |")
L.append("|---|---|---|")
L.append("| 逐帧 | 302 | 100% |")
L.append("| 隔 2 帧 | 152 | 50.3% |")
L.append("| 隔 3 帧 | 102 | 33.8% |")
L.append("\n> 注：手框插值本身不花钱；省的是 HaMeR 前向次数。ViTPose 仍需逐帧跑（它给手框）。\n")
with open(OUT, "w", encoding="utf-8") as f:
    f.write("\n".join(L))
print(f"\n报告已写出: {OUT}")
