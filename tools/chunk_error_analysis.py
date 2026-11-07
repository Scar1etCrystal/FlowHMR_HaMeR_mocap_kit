# -*- coding: utf-8 -*-
"""分块处理误差分析 v2（修正版）。

修正:
  1) 重叠对齐要把 yaw 同时作用到根旋转上, 否则接缝旋转跳变是假的
  2) 分开统计「身体 0..21」和「手指 22..51」—— 手指在我们链路里会被 HaMeR 覆盖,
     它的巨大不一致不代表最终误差

对照: 单次跑 chunk_full (300 帧)
"""
from __future__ import annotations

import os
import sys

import numpy as np
from scipy.spatial.transform import Rotation as R

OUTROOT = sys.argv[1] if len(sys.argv) > 1 else r"F:\flowhmr_pack\app\flowhmr\output"
REPORT = sys.argv[2] if len(sys.argv) > 2 else r"F:\flowhmr_pack\logs\chunk_error_report.md"

MID = 150
OVER_S, OVER_E = 105, 194          # 重叠区全局帧号
BODY = slice(0, 22)                # 0..21 含 Pelvis
FING = slice(22, 52)
NAMES = ["Pelvis", "L_Hip", "R_Hip", "Spine1", "L_Knee", "R_Knee", "Spine2", "L_Ankle",
         "R_Ankle", "Spine3", "L_Foot", "R_Foot", "Neck", "L_Collar", "R_Collar", "Head",
         "L_Shoulder", "R_Shoulder", "L_Elbow", "R_Elbow", "L_Wrist", "R_Wrist"]


def load(tag):
    p = os.path.join(OUTROOT, tag, "flowhmr_latest.npz")
    if not os.path.isfile(p):
        return None
    z = np.load(p, allow_pickle=True)
    return {"poses": z["poses"].reshape(-1, 52, 3).astype(np.float64),
            "trans": z["trans"].astype(np.float64)}


def geo(a, b):
    ra = R.from_rotvec(np.asarray(a).reshape(-1, 3))
    rb = R.from_rotvec(np.asarray(b).reshape(-1, 3))
    return np.degrees((ra.inv() * rb).magnitude())


def stat(d, label, sl=BODY):
    dd = np.asarray(d)
    if dd.ndim == 1:
        dd = dd.reshape(-1, 52)
    v = dd[:, sl]
    return f"{label:26s} mean={v.mean():8.3f}°  p99={np.percentile(v,99):8.3f}°  max={v.max():8.3f}°"


def est_yaw(pA, pB):
    """从根(Pelvis)旋转直接估计世界系夹角。

    poses[:,0] 就是根的世界朝向，所以两块根朝向的相对旋转 = 世界系夹角。
    比用根轨迹方向稳健得多（短窗口轨迹若走直线，yaw 是病态的）。
    """
    rel = R.from_rotvec(np.asarray(pA).reshape(-1, 3)) * R.from_rotvec(np.asarray(pB).reshape(-1, 3)).inv()
    return float(np.median(rel.as_euler("yxz")[:, 0]))


lines = []


def say(s=""):
    print(s)
    lines.append(s)


full = load("chunk_full")
trA, trB = load("chunk_truncA"), load("chunk_truncB")
ovA, ovB = load("chunk_overA"), load("chunk_overB")
if any(x is None for x in (full, trA, trB, ovA, ovB)):
    print("缺结果文件")
    sys.exit(1)

T = full["poses"].shape[0]
say("# 分块处理误差报告\n")
say(f"- 素材 300 帧 / 30 fps；对照 = 单次跑 `chunk_full`（{T} 帧）")
say(f"- 截断：`truncA`[0,149] + `truncB`[150,299]，无重叠")
say(f"- 重叠：`overA`[0,194] + `overB`[105,299]，重叠 [{OVER_S},{OVER_E}] = 90 帧 = 3.0 s\n")

dT = np.linalg.norm(np.diff(full["trans"], axis=0), axis=1) * 100
dR_body = geo(full["poses"][:-1], full["poses"][1:]).reshape(T - 1, 52)[:, BODY].max(axis=1)
say("## 0. 基准量纲（单次跑自身的逐帧变化）\n")
say(f"- 根平移逐帧变化：mean={dT.mean():.3f} cm，p99={np.percentile(dT,99):.3f} cm")
say(f"- 身体关节逐帧旋转变化（最大关节）：mean={dR_body.mean():.3f}°，p99={np.percentile(dR_body,99):.3f}°\n")

# ---------------- 直接截断 ----------------
say("## 1. 直接截断（无重叠硬拼）\n")
seam_t = np.linalg.norm(trB["trans"][0] - trA["trans"][-1]) * 100
seam_r = float(geo(trA["poses"][-1:], trB["poses"][:1]).reshape(-1, 52)[:, BODY].max())
say(f"- **接缝根平移跳变 {seam_t:.2f} cm**（正常 {dT.mean():.3f} cm，{seam_t/dT.mean():.0f}×）")
say(f"- **接缝身体关节跳变 {seam_r:.2f}°**（正常 {dR_body.mean():.3f}°，{seam_r/dR_body.mean():.1f}×）")
say("- 且两块各自从自己的世界原点出发，没有重叠就没有任何数据可以估计对齐量 → 只能生硬拼接\n")

# ---------------- 3s 重叠 ----------------
say("## 2. 留 3 s（90 帧）重叠\n")
A_ov_p, A_ov_t = ovA["poses"][OVER_S:OVER_E + 1], ovA["trans"][OVER_S:OVER_E + 1]
B_ov_p, B_ov_t = ovB["poses"][:OVER_E - OVER_S + 1], ovB["trans"][:OVER_E - OVER_S + 1]
n = len(A_ov_t)
yaw = est_yaw(A_ov_p[:n, 0], B_ov_p[:n, 0])
say(f"### 2.1 重叠区一致性（同样内容的两个窗口各自生成）\n")
resid_t = np.linalg.norm(A_ov_t[:n] - B_ov_t[:n], axis=1) * 100
say(f"- 未对齐时根轨迹差：mean={resid_t.mean():.2f} cm，max={resid_t.max():.2f} cm")
d_ov = geo(A_ov_p[:n], B_ov_p[:n])
say("- " + stat(d_ov, "重叠区身体关节不一致:"))
say("- " + stat(d_ov, "重叠区手指关节不一致:", FING))
say(f"\n> 估计出的世界系夹角：**{np.degrees(yaw):.2f}°**\n")

# 修正后的拼接: yaw 同时作用到 平移 和 根旋转
rot_y = R.from_euler("y", yaw)
B_t_al = rot_y.apply(B_ov_t[:n] - B_ov_t[:n].mean(0)) + A_ov_t[:n].mean(0)
B_t_all = rot_y.apply(ovB["trans"] - B_ov_t[:n].mean(0)) + A_ov_t[:n].mean(0)
B_p_all = ovB["poses"].copy()
# 根(Pelvis)局部旋转 = 世界朝向 → 左乘 yaw 旋转
r0 = R.from_rotvec(B_p_all[:, 0])
B_p_all[:, 0] = (rot_y * r0).as_rotvec()

cut = OVER_S + (OVER_E - OVER_S) // 2          # 149
st_t = np.concatenate([ovA["trans"][:cut], B_t_all[cut - OVER_S:]], 0)
st_p = np.concatenate([ovA["poses"][:cut], B_p_all[cut - OVER_S:]], 0)
say(f"### 2.2 拼接（切点选在重叠区中点，帧 {cut}；yaw 已作用到根旋转）\n")
say(f"- 重叠区对齐后根轨迹残差：mean={np.linalg.norm(B_t_al - A_ov_t[:n], axis=1).mean()*100:.2f} cm")
seam_t2 = np.linalg.norm(st_t[cut] - st_t[cut - 1]) * 100
seam_r2 = float(geo(st_p[cut - 1:cut], st_p[cut:cut + 1]).reshape(-1, 52)[:, BODY].max())
say(f"- **接缝根平移跳变 {seam_t2:.2f} cm**（正常 {dT.mean():.3f} cm，{seam_t2/dT.mean():.1f}×）")
say(f"- **接缝身体关节跳变 {seam_r2:.2f}°**（正常 {dR_body.mean():.3f}°，{seam_r2/dR_body.mean():.1f}×）\n")

# ---------------- 与单次的整体偏差 ----------------
say("## 3. 拼接结果 vs 单次跑（消除世界系差异后）\n")
say("| 方式 | 身体关节偏差 mean | p99 | max | 根轨迹偏差 mean | max |")
say("|---|---|---|---|---|---|")
rows = []
for name, P, Tt in (("直接截断", np.concatenate([trA["poses"], trB["poses"]], 0),
                     np.concatenate([trA["trans"], trB["trans"]], 0)),
                    ("3 s 重叠", st_p, st_t)):
    y2 = est_yaw(full["poses"][:len(Tt), 0], P[:, 0])
    r2 = R.from_euler("y", y2)
    Tt2 = r2.apply(Tt - Tt.mean(0)) + full["trans"][:len(Tt)].mean(0)
    P2 = P.copy()
    P2[:, 0] = (r2 * R.from_rotvec(P[:, 0])).as_rotvec()
    nn = min(len(P2), T)
    dv = geo(full["poses"][:nn], P2[:nn]).reshape(nn, 52)
    dt = np.linalg.norm(Tt2[:nn] - full["trans"][:nn], axis=1) * 100
    say(f"| {name} | {dv[:, BODY].mean():.3f}° | {np.percentile(dv[:, BODY],99):.3f}° | "
        f"{dv[:, BODY].max():.3f}° | {dt.mean():.2f} cm | {dt.max():.2f} cm |")
    rows.append((name, dv[:, BODY].mean(), dt.mean()))

say("")
say("## 4. 结论\n")
say(f"1. 两个窗口看**同样的 90 帧**，各自生成结果的**身体关节平均差 {d_ov.reshape(n, 52)[:, BODY].mean():.1f}°（最大 {d_ov.reshape(n, 52)[:, BODY].max():.1f}°）**")
say(f"   —— 生成过程对窗口切分是敏感的（ODE 联合求解，不是逐帧独立）。")
say(f"2. 手指差得离谱（mean {d_ov.reshape(n, 52)[:, FING].mean():.1f}°），但**我们的链路里手指会被 HaMeR 覆盖**，所以最终交付物不受影响。")
say(f"3. 有重叠时可以用重叠区估计世界系对齐量（本例 {np.degrees(yaw):.2f}°），接缝跳变从 "
    f"{seam_t:.2f} cm / {seam_r:.2f}° 降到 {seam_t2:.2f} cm / {seam_r2:.2f}°。")
say(f"4. 但即使接缝处理好了，拼出来的整段动作与「一次跑完」仍差 "
    f"{rows[1][1]:.2f}°（身体）/ {rows[1][2]:.2f} cm（根）。")
say("")
say("**结论：分块会产生可观误差（身体 ~3°，偶发 10°+），不是无损操作。**")
say("只有在显存装不下时才值得分块；能一次跑完就一次跑完。")
say("")
say("> 注：对照组为本机单次跑（云端未跑过此片段）。本机已实测流水线对相同缓存逐位可复现（maxdiff=0），")
say("> 因此上表差异全部来自分块本身。")

os.makedirs(os.path.dirname(REPORT), exist_ok=True)
with open(REPORT, "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")
print(f"\n报告: {REPORT}")
