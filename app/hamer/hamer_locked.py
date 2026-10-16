"""锁定主角的 HaMeR 视频运行器（阶段 B，云端 Linux）。

与用户 HAMER/hamer_video.py 的差别（其余逐行沿用原版函数：裁剪、平滑、导出格式）：
- 不检人：人框 = FastSAM3DBody 主角跟踪框（main_track.npz），ViTPose 只对主角跑 → 手一定是主角的；
- 每帧都检测（原版隔 2 帧检测、中间帧插值手框）；手关键点都 ≤0.5 的帧照原版插值手框，但记下"手框是插值来的"；
- 导出三份：
  ① <stem>_hamer_data.pkl：与原版同格式、同样 savgol 平滑，合并脚本直接可用；
  ② <stem>_hamer_raw.npz：逐帧未平滑结果 + 信号（主角 ViTPose 133 点逐点置信度、手框及来源、HaMeR 2D 关键点、
     多裁剪一致性 TTA）；
  ③ --others：画面里其他人的手（ViTPose + HaMeR），用来核对旧链路哪些帧拿了别人的手；
  ④ <stem>_hamer_data_i2.pkl：对照，锁主角但照原版隔 2 帧检测。

在 HAMER 代码根目录下运行（_DATA、third-party/ViTPose 按当前目录找）：
    python hamer_locked.py --video V --main_track main_track.npz --out_dir D [--track_raw track_raw.npz --others]
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path


def _cgroup_cores():
    """容器实际可用核数（AutoDL：nproc 报 128，cgroup 只给 12 核）。"""
    try:
        q, p = open("/sys/fs/cgroup/cpu.max").read().split()[:2]
        return max(1, int(int(q) / int(p))) if q != "max" else os.cpu_count()
    except Exception:
        return os.cpu_count() or 1


# torch / OpenCV / BLAS 默认按 nproc 开线程；超过 cgroup 配额会被限流到几乎停住（同时跑两段时实测卡在第 0 帧）。
# 在 import numpy/torch 之前定线程数：HAMER_THREADS 显式给，否则 = 配额核数 / HAMER_PROCS（同时跑几个进程）
THREADS = int(os.environ.get("HAMER_THREADS") or max(1, _cgroup_cores() // int(os.environ.get("HAMER_PROCS", "1"))))
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, str(THREADS))

sys.path.insert(0, os.getcwd())
import hamer_video as HV  # noqa: E402  原版：process_patch / smooth_rotations / decode_* / LEFT_MIRROR

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy.interpolate import interp1d  # noqa: E402
from scipy.signal import medfilt, savgol_filter  # noqa: E402

from hamer.models import DEFAULT_CHECKPOINT, load_hamer  # noqa: E402
from hamer.utils.renderer import cam_crop_to_full  # noqa: E402
from vitpose_model import ViTPoseModel  # noqa: E402
import hamer.models.hamer as _hamer_model  # noqa: E402


def _no_mesh_renderer(*_a, **_k):
    # 模型构造时会建一个训练可视化用的 pyrender 离屏渲染器；云端无显示环境建不起来（报 AttributeError，原版只接
    # ImportError/OSError）。推理用不到它 → 走原版自带的"渲染器不可用"分支（mesh_renderer = None），数值不受影响
    raise OSError("headless: mesh renderer disabled")


_hamer_model.MeshRenderer = _no_mesh_renderer

# 多裁剪一致性：(缩放, x 平移, y 平移)，平移按手框边长计；第 0 个 = 原版裁剪
TTA = [(1.0, 0.0, 0.0), (0.85, 0.0, 0.0), (1.15, 0.0, 0.0), (1.0, 0.1, 0.0), (1.0, -0.1, 0.0), (1.0, 0.0, 0.1), (1.0, 0.0, -0.1)]
BOX_NONE, BOX_DET, BOX_INTERP = 0, 1, 2
HAND_SLICES = ((0, slice(-42, -21)), (1, slice(-21, None)))       # 0 左 1 右（与原版导出顺序相同）


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True)
    p.add_argument("--main_track", required=True, help="FastSAM3DBody main_track.npz（bbox_xyxy、detected）")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--track_raw", default="", help="FastSAM3DBody track_raw.npz（所有人的 YOLO 框），--others 用")
    p.add_argument("--others", action="store_true", help="也跑画面里其他人的手")
    p.add_argument("--tta", type=int, default=len(TTA), help="多裁剪个数（1 = 只跑原版裁剪）")
    p.add_argument("--batch_size", type=int, default=48)
    p.add_argument("--window", type=int, default=11)
    p.add_argument("--max_frames", type=int, default=0, help="调试：只跑前 N 帧")
    p.add_argument("--intervals", default="2",
                   help="对照组：隔 N 帧检测+中间插值，逗号分隔（如 2,3）；主结果始终是逐帧")
    return p.parse_args()


def hand_box(kps, min_side=8.0):
    """原版规则：该手 21 点里置信度 >0.5 的点的外接框；没有这样的点 → None。
    只多一条：框边长至少 min_side 像素（原版只有 1 个点过线时框退化成一个点，裁剪尺寸为 0）。"""
    if kps[:, 2].max() <= 0.5:
        return None
    v = kps[:, 2] > 0.5
    b = np.array([kps[v, 0].min(), kps[v, 1].min(), kps[v, 0].max(), kps[v, 1].max()], np.float32)
    for i in (0, 1):
        if b[i + 2] - b[i] < min_side:
            c = (b[i] + b[i + 2]) / 2
            b[i], b[i + 2] = c - min_side / 2, c + min_side / 2
    return b


def perturb(box, s, dx, dy):
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    w, h = (box[2] - box[0]) * s, (box[3] - box[1]) * s
    L = max(box[2] - box[0], box[3] - box[1])
    cx, cy = cx + dx * L, cy + dy * L
    return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], np.float32)


def iou(a, b):
    x0, y0, x1, y1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


class HamerBatcher:
    """攒批跑 HaMeR；每个任务带回调键，结果按键存。"""

    def __init__(self, model, cfg, device, focal_length, batch_size):
        self.model, self.cfg, self.device, self.focal, self.bs = model, cfg, device, focal_length, batch_size
        self.items, self.out = [], {}

    def add(self, key, frame, box, is_right):
        self.items.append((key, HV.process_patch(frame, box, is_right, self.cfg)))
        if len(self.items) >= self.bs:
            self.flush()

    def flush(self):
        if not self.items:
            return
        metas = [m for _, m in self.items]
        batch = {
            "img": torch.stack([torch.from_numpy(m["img"]) for m in metas]).float().to(self.device),
            "right": torch.from_numpy(np.array([m["right"] for m in metas], np.float32)).to(self.device),
            "box_center": torch.from_numpy(np.stack([m["box_center"] for m in metas]).astype(np.float32)).to(self.device),
            "box_size": torch.from_numpy(np.array([m["box_size"] for m in metas], np.float32)).to(self.device),
            "img_size": torch.from_numpy(np.stack([m["img_size"] for m in metas]).astype(np.float32)).to(self.device),
        }
        with torch.no_grad():
            out = self.model(batch)
        pred_cam = out["pred_cam"].clone()
        pred_cam[:, 1] *= 2 * batch["right"] - 1
        cam_t = cam_crop_to_full(pred_cam, batch["box_center"], batch["box_size"], batch["img_size"], self.focal).cpu().numpy()
        kp2 = out["pred_keypoints_2d"].cpu().numpy()                     # 裁剪归一化坐标（以裁剪中心为原点，单位 = 裁剪边长）
        for i, (key, m) in enumerate(self.items):
            sgn = 1.0 if m["right"] > 0.5 else -1.0                     # 左手是在翻转图里算的
            full = np.stack([m["box_center"][0] + sgn * kp2[i, :, 0] * m["box_size"],
                             m["box_center"][1] + kp2[i, :, 1] * m["box_size"]], -1)
            self.out[key] = {
                "pose": out["pred_mano_params"]["hand_pose"][i].cpu().numpy(),
                "orient": out["pred_mano_params"]["global_orient"][i].cpu().numpy(),
                "betas": out["pred_mano_params"]["betas"][i].cpu().numpy(),
                "cam_t": cam_t[i], "kp2d": full.astype(np.float32),
                "kp3d": out["pred_keypoints_3d"][i].cpu().numpy(),
            }
        self.items.clear()


def export_like_original(tasks, num_frames, window):
    """原版 [3/4] 平滑 + [4/4] 导出，逐行照抄（smplx_model_path 为空 → hand_pose_abs_flat 全零，与插件默认一致）。"""
    for hand_type in (0, 1):
        ht = sorted([t for t in tasks if t["is_right"] == hand_type], key=lambda x: x["frame_idx"])
        if len(ht) < window:
            continue
        poses = np.array([t["results"]["pose"] for t in ht])
        orients = np.array([t["results"]["orient"] for t in ht])
        cams = np.array([t["results"]["cam_t"] for t in ht])
        s_poses = HV.smooth_rotations(poses, window)
        s_orients = HV.smooth_rotations(orients, window)
        s_cams = np.zeros_like(cams)
        for dim in range(3):
            s_cams[:, dim] = medfilt(cams[:, dim], kernel_size=5)
        s_cams = savgol_filter(s_cams, window, 3, axis=0)
        for idx, t in enumerate(ht):
            t["results"] = dict(t["results"], pose=s_poses[idx], orient=s_orients[idx], cam_t=s_cams[idx])
    ex = {
        "global_orient": np.zeros((num_frames, 2, 3, 3), dtype=np.float32),
        "hand_pose": np.zeros((num_frames, 2, 15, 3, 3), dtype=np.float32),
        "global_orient_valid": np.zeros((num_frames, 2), dtype=np.uint8),
        "hand_pose_abs_flat": np.zeros((num_frames, 2, 15, 3, 3), dtype=np.float32),
        "global_orient_abs_lr": np.zeros((num_frames, 2, 3, 3), dtype=np.float32),
    }
    for t in tasks:
        f, res = t["frame_idx"], t["results"]
        _, om = HV.decode_single_rot(res["orient"])
        _, pm = HV.decode_hand_pose(res["pose"])
        h = 0 if t["is_right"] == 0 else 1
        ex["global_orient"][f, h] = om
        ex["hand_pose"][f, h] = pm
        ex["global_orient_valid"][f, h] = 1
        ex["global_orient_abs_lr"][f, h] = om if t["is_right"] == 1 else HV.LEFT_MIRROR @ om @ HV.LEFT_MIRROR
    return ex


def main():
    a = parse_args()
    t0 = time.time()
    torch.set_num_threads(THREADS)
    cv2.setNumThreads(THREADS)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(a.video).stem
    model, cfg = load_hamer(DEFAULT_CHECKPOINT)
    model = model.to(device).eval()
    pose_model = ViTPoseModel(device)
    print(f"models loaded {time.time() - t0:.0f}s on {device}", flush=True)

    mt = np.load(a.main_track)
    mbox, mdet = mt["bbox_xyxy"].astype(np.float32), mt["detected"].astype(bool)
    others_by_frame = {}
    if a.others and a.track_raw:
        tr = np.load(a.track_raw)
        for f, b, c, tid in zip(tr["frame"], tr["xyxy"], tr["conf"], tr["tid"]):
            others_by_frame.setdefault(int(f), []).append((np.asarray(b, np.float32), float(c), int(tid)))

    cap = cv2.VideoCapture(a.video)
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))              # 原版按容器帧数开数组（可能比能解出的多 1 帧）
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if a.max_frames:
        num_frames = min(num_frames, a.max_frames)
    focal = cfg.EXTRA.FOCAL_LENGTH / cfg.MODEL.IMAGE_SIZE * max(W, H)

    # ---------- 第 1 遍：主角（和其他人）的 ViTPose ----------
    vit = np.zeros((num_frames, 133, 3), np.float32)
    person_box = np.full((num_frames, 4), np.nan, np.float32)
    person_src = np.zeros(num_frames, np.uint8)                       # 0 无 1 跟踪检测到 2 跟踪插值/保持
    det_box = {0: {}, 1: {}}
    others = []                                                        # (frame, tid, side, box(4), kps(21,3))
    n_read = 0
    for f in range(num_frames):
        ok, frame = cap.read()
        if not ok:
            break
        n_read += 1
        boxes = []
        if f < len(mbox) and np.all(np.isfinite(mbox[f])) and (mbox[f][2] - mbox[f][0]) > 4:
            boxes.append(np.r_[mbox[f], 1.0])
            person_box[f], person_src[f] = mbox[f], 1 if mdet[f] else 2
        oth = []
        if boxes:
            for b, c, tid in others_by_frame.get(f, []):
                if iou(b, mbox[f]) < 0.3:
                    oth.append((b, tid))
                    boxes.append(np.r_[b, 1.0])
        if not boxes:
            continue
        res = pose_model.predict_pose(frame[:, :, ::-1], [np.stack(boxes).astype(np.float32)])
        vit[f] = res[0]["keypoints"]
        for h, sl in HAND_SLICES:
            b = hand_box(vit[f][sl])
            if b is not None:
                det_box[h][f] = b
        for (b, tid), r in zip(oth, res[1:]):
            for h, sl in HAND_SLICES:
                hb = hand_box(r["keypoints"][sl])
                if hb is not None:
                    others.append((f, tid, h, hb, r["keypoints"][sl].astype(np.float32)))
        if f % 200 == 0:
            print(f"[1/3] vitpose {f}/{num_frames} {time.time() - t0:.0f}s", flush=True)

    # 手框：检测到的帧用检测框；首末检测之间缺的帧照原版线性插值（记为 INTERP）
    boxes = np.zeros((num_frames, 2, 4), np.float32)
    box_src = np.zeros((num_frames, 2), np.uint8)
    tasks = []
    for h in (0, 1):
        fr = sorted(det_box[h])
        if not fr:
            continue
        fn = interp1d(fr, [det_box[h][f] for f in fr], axis=0, kind="linear", fill_value="extrapolate")
        for f in range(min(fr), max(fr) + 1):
            boxes[f, h] = det_box[h][f] if f in det_box[h] else fn(f)
            box_src[f, h] = BOX_DET if f in det_box[h] else BOX_INTERP
            tasks.append({"frame_idx": f, "bbox": boxes[f, h], "is_right": h, "results": {}})
    # 对照：锁主角、但隔 N 帧检测（只用 f % N == 0 的检测框，其余线性插值）
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
              f"{len(_ts) - _nd} extra HaMeR calls", flush=True)
    print(f"[1/3] done {time.time() - t0:.0f}s; hand boxes detected L {int((box_src[:, 0] == 1).sum())} "
          f"R {int((box_src[:, 1] == 1).sum())}, interpolated L {int((box_src[:, 0] == 2).sum())} R {int((box_src[:, 1] == 2).sum())}; "
          f"others' hands {len(others)}", flush=True)

    # ---------- 第 2 遍：HaMeR（主角 × 多裁剪 + 其他人） ----------
    K = max(1, min(a.tta, len(TTA)))
    hb = HamerBatcher(model, cfg, device, focal, a.batch_size)
    by_frame = {}
    for t in tasks:
        by_frame.setdefault(t["frame_idx"], []).append(t)
    cmp_by_frame = {}
    for _N, _ts in cmp_tasks.items():
        _d = {}
        for t in _ts:
            if not t["reuse"]:
                _d.setdefault(t["frame_idx"], []).append(t)
        cmp_by_frame[_N] = _d
    oth_by_frame = {}
    for i, o in enumerate(others):
        oth_by_frame.setdefault(o[0], []).append(i)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    for f in range(n_read):
        ok, frame = cap.read()
        if not ok:
            break
        for t in by_frame.get(f, []):
            for k in range(K):
                hb.add(("m", f, t["is_right"], k), frame, perturb(t["bbox"], *TTA[k]) if k else t["bbox"], t["is_right"])
        for _N, _d in cmp_by_frame.items():
            for t in _d.get(f, []):
                hb.add(("cmp", _N, f, t["is_right"]), frame, t["bbox"], t["is_right"])
        for i in oth_by_frame.get(f, []):
            hb.add(("o", i), frame, others[i][3], others[i][2])
        if f % 200 == 0:
            print(f"[2/3] hamer {f}/{num_frames} {time.time() - t0:.0f}s", flush=True)
    hb.flush()
    cap.release()

    # ---------- 第 3 遍：整理导出 ----------
    F = num_frames
    raw = {"orient": np.zeros((F, 2, 3, 3), np.float32), "pose": np.zeros((F, 2, 15, 3, 3), np.float32),
           "betas": np.zeros((F, 2, 10), np.float32), "cam_t": np.zeros((F, 2, 3), np.float32),
           "kp2d": np.zeros((F, 2, 21, 2), np.float32), "kp3d": np.zeros((F, 2, 21, 3), np.float32),
           "tta_orient": np.zeros((F, 2, K, 3, 3), np.float32), "tta_pose": np.zeros((F, 2, K, 15, 3, 3), np.float32)}
    for t in tasks:
        f, h = t["frame_idx"], t["is_right"]
        r = hb.out[("m", f, h, 0)]
        t["results"] = {k: r[k] for k in ("pose", "orient", "betas", "cam_t")}
        for k in ("orient", "pose", "betas", "cam_t", "kp2d", "kp3d"):
            raw[k][f, h] = np.asarray(r[k]).reshape(raw[k][f, h].shape)
        for k in range(K):
            rk = hb.out[("m", f, h, k)]
            raw["tta_orient"][f, h, k] = np.asarray(rk["orient"]).reshape(3, 3)
            raw["tta_pose"][f, h, k] = np.asarray(rk["pose"]).reshape(15, 3, 3)
    ex = export_like_original(tasks, num_frames, a.window)
    with open(out_dir / f"{stem}_hamer_data.pkl", "wb") as fh:
        pickle.dump(ex, fh, protocol=4)
    for _N, _ts in cmp_tasks.items():
        for t in _ts:
            _key = (("m", t["frame_idx"], t["is_right"], 0) if t["reuse"]
                    else ("cmp", _N, t["frame_idx"], t["is_right"]))
            r = hb.out[_key]
            t["results"] = {k: r[k] for k in ("pose", "orient", "betas", "cam_t")}
        _out = out_dir / f"{stem}_hamer_data_i{_N}.pkl"
        with open(_out, "wb") as fh:
            pickle.dump(export_like_original(_ts, num_frames, a.window), fh, protocol=4)
        print(f"[3/3] exported {_out.name}  ({len(_ts)} tasks)", flush=True)
    np.savez_compressed(out_dir / f"{stem}_hamer_raw.npz", vit_kp=vit, person_box=person_box, person_src=person_src,
                        hand_box=boxes, box_src=box_src, tta_params=np.array(TTA[:K], np.float32), img_wh=np.array([W, H]),
                        fps=np.float32(fps), focal=np.float32(focal), **raw)
    if others:
        oo = [hb.out[("o", i)] for i in range(len(others))]
        np.savez_compressed(out_dir / f"{stem}_others.npz",
                            frame=np.array([o[0] for o in others]), tid=np.array([o[1] for o in others]),
                            side=np.array([o[2] for o in others]), box=np.stack([o[3] for o in others]),
                            vit_kp=np.stack([o[4] for o in others]),
                            orient=np.stack([np.asarray(r["orient"]).reshape(3, 3) for r in oo]).astype(np.float32),
                            pose=np.stack([np.asarray(r["pose"]).reshape(15, 3, 3) for r in oo]).astype(np.float32),
                            kp2d=np.stack([r["kp2d"] for r in oo]))
    info = {"video": a.video, "threads": THREADS, "frames": F, "frames_decoded": n_read, "W": W, "H": H, "fps": fps, "tta": K, "others": len(others),
            "seconds": round(time.time() - t0, 1), "torch": torch.__version__, "numpy": np.__version__,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"}
    (out_dir / "run_info.json").write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding="utf-8")
    print("DONE", json.dumps(info, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
