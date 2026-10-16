# -*- coding: utf-8 -*-
"""FlowHaMeR 包补丁器：幂等 / 锚点唯一性断言 / 自动备份 / 语法检查。

用法:
    envs\\python.exe tools\\patch_all.py            # 打补丁
    envs\\python.exe tools\\patch_all.py --check    # 只检查，不写
    envs\\python.exe tools\\patch_all.py --revert   # 从 .orig_pack 还原

补丁清单
  P1 runtime_sam_features.py         Fast 的手部 batch 合并兼容（body 保持 [0]）
  P2 runtime_v2m_postprocess_simple.py  POSTPROCESS_SKIP_SMOOTH 开关（去滤波）
  P3 runtime_vggt_camera.py          新增 release_vggt_model()（显存回收）
  P4 v2m_inference.py                阶段间调用释放（VGGT / SAM3D）
"""
from __future__ import annotations

import argparse
import os
import py_compile
import shutil
import sys
from pathlib import Path

PACK = Path(__file__).resolve().parent.parent
REPO = PACK / "app" / "flowhmr"
U = REPO / "flowhmr" / "utils"
P = REPO / "flowhmr" / "pipeline"

# ---------------------------------------------------------------- patch table
# 每条: (文件, 名字, [(old, new), ...])
PATCHES = []

# ---- P1: Fast-SAM-3D-Body 手部 batch 合并兼容 -------------------------------
PATCHES.append((U / "runtime_sam_features.py", "P1-guard", [(
    "        if len(self._body_tokens) < 1 or len(self._hand_tokens) < 2:",
    "        if len(self._body_tokens) < 1 or len(self._hand_tokens) < 1:",
)]))

PATCHES.append((U / "runtime_sam_features.py", "P1-select", [(
    """        body_token = self._body_tokens[0]
        lhand_token = self._hand_tokens[0]
        rhand_token = self._hand_tokens[1]""",
    """        # Fast-SAM-3D-Body merges left+right hand into ONE decoder call whose
        # batch dim is [left, right]; upstream official calls the hand decoder
        # twice.  body_token stays [0] = the FIRST pass, which is what upstream
        # uses (verified bit-identical to official: cos 1.0000, maxdiff 3.3e-06).
        body_token = self._body_tokens[0]
        if self._hand_tokens[0].shape[0] >= 2:
            _hb = self._hand_tokens[0]
            lhand_token = _hb[0:1]
            rhand_token = _hb[1:2]
        elif len(self._hand_tokens) >= 2:
            lhand_token = self._hand_tokens[0]
            rhand_token = self._hand_tokens[1]
        else:
            raise RuntimeError(
                "cannot resolve lhand/rhand from "
                f"{len(self._hand_tokens)} hand token call(s)"
            )""",
)]))

# ---- P2: 跳过后处理的时间滤波 ------------------------------------------------
# 先补 import os（该文件原本没导入）
PATCHES.append((U / "runtime_v2m_postprocess_simple.py", "P2a-import-os", [(
    "import torch\nimport numpy as np\nfrom ..pipeline.lbfgs import",
    "import os\nimport torch\nimport numpy as np\nfrom ..pipeline.lbfgs import",
)]))

PATCHES.append((U / "runtime_v2m_postprocess_simple.py", "P2-skip-smooth", [(
    "        output = _smooth_motion(output)",
    """        if os.environ.get("POSTPROCESS_SKIP_SMOOTH", "0") == "1":
            print("  [postprocess] skipping step2 _smooth_motion")
        else:
            output = _smooth_motion(output)""",
)]))

# ---- P3: VGGT 显存回收函数 ---------------------------------------------------
_VGGT_RELEASE = '''

def release_vggt_model() -> None:
    """释放模块级 VGGT 缓存（上游从不释放，~4.4 GB 会一直占着显存）。

    相机阶段结束后调用即可：后续 extract_vggt_camera() 会命中
    *_vggt_camera.npz 缓存直接 early-return，不再需要模型。
    """
    import gc

    n = len(_VGGT_MODEL_CACHE)
    _VGGT_MODEL_CACHE.clear()
    gc.collect()
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    print(f"  [vram] released VGGT model cache ({n} entr{'y' if n == 1 else 'ies'})")
'''
PATCHES.append((U / "runtime_vggt_camera.py", "P3-release-fn", [(
    "\ndef extract_vggt_camera(",
    _VGGT_RELEASE + "\ndef extract_vggt_camera(",
)]))

# ---- P4: 阶段间释放 ----------------------------------------------------------
_UNLOAD_VGGT = '''        # ---- VRAM: 相机用完就放（Step 4 会命中 npz 缓存, 不再需要模型）----
        if os.environ.get("FLOWHMR_UNLOAD_BETWEEN_STAGES", "1") == "1":
            try:
                from flowhmr.utils.runtime_vggt_camera import release_vggt_model
                release_vggt_model()
            except Exception as _e:  # 释放失败不能影响主流程
                print(f"  [vram] VGGT release skipped: {_e}")

        # ==== Step 3: SAM feature extraction ===='''
PATCHES.append((P / "v2m_inference.py", "P4-unload-vggt", [(
    "        # ==== Step 3: SAM feature extraction ====",
    _UNLOAD_VGGT,
)]))

_UNLOAD_SAM = '''        # ---- VRAM: 生成前丢掉 SAM3D 提取器(2.1 GB) ----
        if os.environ.get("FLOWHMR_UNLOAD_BETWEEN_STAGES", "1") == "1" \\
                and self.sam_extractor is not None:
            try:
                self.sam_extractor = None
                import gc as _gc
                _gc.collect()
                torch.cuda.empty_cache()
                print("  [vram] released SAM3D extractor")
            except Exception as _e:
                print(f"  [vram] SAM release skipped: {_e}")

        # ==== Step 4: Generation ===='''
PATCHES.append((P / "v2m_inference.py", "P4-unload-sam", [(
    "        # ==== Step 4: Generation ====",
    _UNLOAD_SAM,
)]))


# ---- P4b: 惰性重建 SAM3D（多视频支持）--------------------------------------
# P4-unload-sam 会把 self.sam_extractor 置空；入口脚本支持一次传多个视频，
# 第二个视频若特征未缓存就会崩在 NoneType.hasattr。这里在 Step 3 按需重建。
_RELOAD_SAM = '''        print(f"\\n  === Step 3: SAM Feature ===")
        _pcb(StageName.FEATURE_EXTRACT, 0, "Starting SAM extraction")

        # 上一个视频可能已按 FLOWHMR_UNLOAD_BETWEEN_STAGES 释放了提取器 → 按需重建
        if self.sam_extractor is None:
            from flowhmr.utils.runtime_sam_features import build_sam3d_extractor
            print("  [vram] SAM3D extractor was released, reloading...")
            _dev = self.device
            if isinstance(_dev, str):
                _dev = torch.device(_dev)
            self.sam_extractor = build_sam3d_extractor(_dev)
'''
PATCHES.append((P / "v2m_inference.py", "P4b-reload-sam", [(
    '        print(f"\\n  === Step 3: SAM Feature ===")\n'
    '        _pcb(StageName.FEATURE_EXTRACT, 0, "Starting SAM extraction")\n',
    _RELOAD_SAM,
)]))


# ---- P5: DINOv3 离线加载（包内 hub 目录优先）--------------------------------
PATCHES.append((REPO / "third_party" / "sam-3d-body" / "sam_3d_body" / "models"
                / "backbones" / "dinov3.py", "P5-dinov3-offline", [(
    '        # Prefer local cache to avoid network requests\n'
    '        local_cache = os.path.expanduser("~/.cache/torch/hub/facebookresearch_dinov3_main")',
    '        # Prefer an explicit repo dir (portable pack), then the user cache.\n'
    '        # Upstream only reads ~/.cache, which forces a GitHub fetch when absent.\n'
    '        local_cache = os.environ.get("DINOV3_LOCAL_REPO") or os.path.expanduser(\n'
    '            "~/.cache/torch/hub/facebookresearch_dinov3_main"\n'
    '        )',
)]))


def backup(path: Path) -> Path:
    bak = path.with_suffix(path.suffix + ".orig_pack")
    if not bak.exists():
        shutil.copy2(path, bak)
        print(f"    backup -> {bak.name}")
    return bak


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="只检查锚点，不写文件")
    ap.add_argument("--revert", action="store_true", help="从 .orig_pack 还原")
    a = ap.parse_args()

    if a.revert:
        n = 0
        for path in {p[0] for p in PATCHES}:
            bak = path.with_suffix(path.suffix + ".orig_pack")
            if bak.exists():
                shutil.copy2(bak, path)
                print(f"  reverted {path.name}")
                n += 1
        print(f"--- reverted {n} file(s)")
        return 0

    ok = True
    for path, name, subs in PATCHES:
        if not path.exists():
            print(f"  ✗ [{name}] 文件不存在: {path}")
            ok = False
            continue
        src = path.read_text(encoding="utf-8")
        if "def _stage_memory_event(" in src and name in {
            "P4-unload-vggt", "P4-unload-sam", "P4b-reload-sam"
        }:
            print(f"  = [{name}] superseded by staged residency patch, skip")
            continue
        todo = []
        for old, new in subs:
            if new in src:
                continue                      # 已打过
            cnt = src.count(old)
            if cnt != 1:
                print(f"  ✗ [{name}] 锚点出现 {cnt} 次(需恰好 1 次) in {path.name}")
                ok = False
                continue
            todo.append((old, new))

        if not todo:
            print(f"  = [{name}] 已是最新，跳过")
            continue
        if a.check:
            print(f"  ? [{name}] 可打 ({len(todo)} 处) in {path.name}")
            continue

        backup(path)
        for old, new in todo:
            src = src.replace(old, new, 1)
        path.write_text(src, encoding="utf-8")
        try:
            py_compile.compile(str(path), doraise=True)
            print(f"  ✓ [{name}] 已打 ({len(todo)} 处) + 语法通过 -> {path.name}")
        except py_compile.PyCompileError as e:
            print(f"  ✗ [{name}] 语法错误, 回滚: {e}")
            shutil.copy2(path.with_suffix(path.suffix + ".orig_pack"), path)
            ok = False

    print("--- OK" if ok else "--- 有问题，见上")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
