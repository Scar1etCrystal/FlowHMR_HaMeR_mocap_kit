# -*- coding: utf-8 -*-
"""FlowHaMeR 包环境自检。

用法（在包根目录）:
    envs\\python.exe tools\\check_env.py            # 依赖 + 文件 + CUDA
    envs\\python.exe tools\\check_env.py --imports  # 只跑依赖导入检查
    envs\\python.exe tools\\check_env.py --smoke    # 额外加载 SAM3D 真跑一帧（吃显存）

退出码 0 = 全过。
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

PACK = Path(__file__).resolve().parent.parent
REPO = PACK / "app" / "flowhmr"
SAM3D = REPO / "third_party" / "sam-3d-body"
VGGT = REPO / "third_party" / "vggt-omega"
YOLOX = REPO / "third_party" / "YOLOX"
HAMER = PACK / "app" / "hamer"

# 依赖 -> 用途
DEPS = [
    ("torch", "核心"),
    ("torchvision", "HaMeR / YOLOX"),
    ("numpy", "核心"),
    ("scipy", "核心"),
    ("cv2", "视频 IO"),
    ("einops", "MM-DiT"),
    ("yaml", "配置"),
    ("omegaconf", "配置"),
    ("tqdm", "进度"),
    ("matplotlib", "可视化"),
    ("imageio", "视频 IO"),
    ("joblib", "FlowHMR 依赖"),
    ("accelerate", "FlowHMR 依赖"),
    ("torchdiffeq", "ODE 求解"),
    ("supervision", "ByteTrack 跟踪"),
    ("imageio_ffmpeg", "转码 ffmpeg"),
    ("roma", "SAM3D 旋转工具"),
    ("smplx", "body model"),
    ("trimesh", "网格"),
    ("PIL", "图像"),
    ("safetensors", "权重"),
    ("huggingface_hub", "权重下载"),
    ("timm", "骨干"),
    ("fvcore", "detectron2 依赖"),
    ("iopath", "detectron2 依赖"),
]

# 运行时必需的文件
FILES = [
    ("app/flowhmr/run_v2m_demo_generation.py", "FlowHMR 入口"),
    ("app/flowhmr/flowhmr/pipeline/v2m_inference.py", "编排层"),
    ("app/flowhmr/third_party/sam-3d-body/sam_3d_body/__init__.py", "Fast SAM3D"),
    ("app/flowhmr/third_party/vggt-omega/vggt_omega/__init__.py", "VGGT"),
    ("app/flowhmr/third_party/YOLOX/yolox/exp/__init__.py", "YOLOX"),
    ("app/flowhmr/third_party/dinov3_hub/facebookresearch_dinov3_main/hubconf.py", "DINOv3 hub"),
    ("app/flowhmr/ckpts/yolox/yolox_l.pth", "YOLOX 权重"),
    ("app/flowhmr/ckpts/vggt-omega/vggt_omega_1b_512.pt", "VGGT 权重"),
    ("app/flowhmr/ckpts/sam-3d-body-dinov3/model.ckpt", "SAM3D 权重"),
    ("app/flowhmr/ckpts/sam-3d-body-dinov3/assets/mhr_model.pt", "MHR 权重"),
    ("app/flowhmr/checkpoints/flowhmr_latest/flowhmr_latest.ckpt", "FlowHMR 权重"),
    ("app/flowhmr/assets/body_models/smplh/neutral/model.npz", "SMPL-H"),
    ("app/flowhmr/assets/body_models/smpl_neutral_J_regressor.pt", "J_regressor"),
    ("app/hamer/hamer_locked.py", "HaMeR 锁定运行器"),
    ("app/hamer/hamer_video.py", "HaMeR 视频运行器"),
    ("app/hamer/hamer/models/hamer.py", "HaMeR 代码"),
    ("app/hamer/_DATA/hamer_ckpts/checkpoints/hamer.ckpt", "HaMeR 权重"),
    ("app/hamer/_DATA/vitpose_ckpts/vitpose+_huge/wholebody.pth", "ViTPose 权重"),
    ("envs/python.exe", "Python"),
]


def check_deps() -> int:
    bad = []
    print("== 依赖 ==")
    for mod, why in DEPS:
        try:
            spec = importlib.util.find_spec(mod)
        except Exception:
            spec = None
        if spec is None:
            bad.append(mod)
            print(f"  缺失  {mod:20s} ({why})")
    if not bad:
        print(f"  全部 {len(DEPS)} 项通过")
    return len(bad)


def check_imports() -> int:
    """真正 import 一遍关键模块（find_spec 会把坏的 editable 安装判成 OK）。"""
    sys.path.insert(0, str(REPO))
    os.chdir(REPO)
    targets = [
        ("torch", f"torch {__import__('torch').__version__}"),
        ("numpy", f"numpy {__import__('numpy').__version__}"),
        ("supervision", "supervision"),
        ("roma", "roma"),
        ("accelerate", "accelerate"),
        ("torchdiffeq", "torchdiffeq"),
        ("imageio_ffmpeg", "imageio_ffmpeg"),
    ]
    bad = 0
    print("== 关键 import ==")
    for mod, label in targets:
        try:
            __import__(mod)
            print(f"  ok    {label}")
        except Exception as e:
            bad += 1
            print(f"  FAIL  {mod}: {type(e).__name__}: {str(e)[:90]}")
    return bad


def check_files() -> int:
    bad = 0
    print("== 文件 ==")
    for rel, why in FILES:
        p = PACK / rel
        if p.exists():
            print(f"  ok    {rel}")
        else:
            bad += 1
            print(f"  缺失  {rel:60s} ({why})")
    return bad


def check_cuda() -> int:
    print("== CUDA ==")
    try:
        import torch
    except Exception as e:
        print(f"  FAIL  torch 导入失败: {e}")
        return 1
    if not torch.cuda.is_available():
        print("  FAIL  torch.cuda.is_available() = False")
        return 1
    d = torch.cuda.get_device_properties(0)
    print(f"  ok    {d.name}  {d.total_memory / 1024**3:.1f} GB  "
          f"(sm_{d.major}{d.minor}, torch {torch.__version__})")
    if d.total_memory / 1024**3 < 10:
        print("  警告  显存 < 10 GB，长视频可能 OOM")
    return 0


def smoke() -> int:
    """加载 SAM3D 真跑一帧，报耗时与峰值显存。"""
    print("== smoke: 加载 Fast-SAM3D 并跑一帧 ==")
    sys.path.insert(0, str(REPO))
    sys.path.insert(0, str(SAM3D))
    os.chdir(REPO)
    os.environ.setdefault("SKIP_KEYPOINT_PROMPT", "1")
    os.environ.setdefault("IMG_SIZE", "512")
    os.environ.setdefault("LAYER_DTYPE", "fp32")
    os.environ.setdefault("GPU_HAND_PREP", "1")
    os.environ.setdefault("MHR_USE_CUDA_GRAPH", "0")
    os.environ.setdefault("DINOV3_LOCAL_REPO",
                          str(REPO / "third_party" / "dinov3_hub" / "facebookresearch_dinov3_main"))

    import time

    import cv2
    import numpy as np
    import torch

    t0 = time.time()
    from sam_3d_body import SAM3DBodyEstimator, load_sam_3d_body

    model, cfg = load_sam_3d_body(
        checkpoint_path=str(REPO / "ckpts" / "sam-3d-body-dinov3" / "model.ckpt"),
        mhr_path=str(REPO / "ckpts" / "sam-3d-body-dinov3" / "assets" / "mhr_model.pt"),
        device="cuda",
    )
    print(f"  模型加载 {time.time() - t0:.1f}s")
    torch.cuda.reset_peak_memory_stats()

    est = SAM3DBodyEstimator(sam_3d_body_model=model, model_cfg=cfg)
    video = PACK / "inputs" / "example.mp4"
    cap = cv2.VideoCapture(str(video))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        print("  FAIL  读不到示例视频")
        return 1
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    bbox = np.array([[w * 0.2, h * 0.05, w * 0.8, h * 0.98]], np.float32)
    K = torch.tensor([[[w, 0, w / 2], [0, w, h / 2], [0, 0, 1]]], dtype=torch.float32)

    t1 = time.time()
    with torch.no_grad():
        out = est.process_one_image(rgb, bboxes=bbox, bbox_thr=0.5, nms_thr=0.3,
                                    use_mask=False, inference_type="full", cam_int=K)
    dt = time.time() - t1
    peak = torch.cuda.max_memory_allocated() / 1024**3
    print(f"  单帧推理 {dt * 1000:.0f} ms  峰值显存 allocated {peak:.2f} GiB")
    print(f"  输出人物数 {len(out) if out else 0}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--imports", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()

    bad = 0
    if a.imports:
        bad += check_deps() + check_imports()
    else:
        bad += check_deps()
        bad += check_imports()
        bad += check_files()
        bad += check_cuda()
        if a.smoke:
            bad += smoke()

    print()
    print("=== 全部通过 ===" if bad == 0 else f"=== {bad} 项有问题 ===")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
