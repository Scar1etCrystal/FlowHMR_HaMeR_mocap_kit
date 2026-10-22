# 给 AI agent 的部署说明（cu121 / cu128 两种构建，覆盖 20–50 系）

> 这份文件是给**代码 agent** 看的操作说明，不是给人读的介绍文档。
> 所有路径都相对本文件所在目录（包根）。请按顺序执行，不要跳过第 6 节的验证。

## 0. 结论先行

**先确认本包是哪种 torch 构建，两种都能用，只是覆盖面不同：**

```bat
envs\python.exe -c "import torch;print(torch.__version__, torch.cuda.get_arch_list())"
```

- `2.5.1+cu121`（**cu121 构建**）：20 / 30 / 40 系直接可用，且在 Ampere 上比 cu128
  快约 19%（151 帧全链 260 s vs 314.7 s）。**不支持 50 系（sm_120）**，
  遇到 50 系机器按第 2 节换成 cu128。
- `2.11.0+cu128`（**cu128 构建**）：20 / 30 / 40 / 50 系全部覆盖，不需要再换。

确认完直接跳到第 5 节跑起来，第 6 节做验证。

## 1. 已确认的事实（不要重复验证）

- 自带环境：Python 3.11.4，torch 是 cu121 或 cu128 构建之一（见第 0 节自检）
- cu121 构建：`torch 2.5.1+cu121` / `torchvision 0.20.1+cu121`，cuDNN 9.1，
  arch list `['sm_50','sm_60','sm_61','sm_70','sm_75','sm_80','sm_86','sm_90']`（**无** sm_120）
- cu128 构建：`torch 2.11.0+cu128` / `torchvision 0.26.0+cu128`，cuDNN 9.19，
  arch list `['sm_75','sm_80','sm_86','sm_90','sm_100','sm_120']`（**含** sm_120）
- 40 系是 sm_89，两种构建都没有，会走同大版本 sm_86 的 cubin
  （NVIDIA 保证同大版本内次版本向上二进制兼容）
- 两种构建都在 RTX 3060（sm_86）上完整跑通过全链，FBX 重导入结果一致
- 包内 `envs\` 自带 pip（26.1.1），用它装包，不要用系统 Python

## 2. 需要 cu128 时（50 系 / sm_120）

只有第 0 节自检发现 arch list 里**没有 `sm_120`** 时才需要做这一步。

```bat
envs\python.exe -m pip install --index-url https://download.pytorch.org/whl/cu128 torch torchvision
```

- 必须 torch >= 2.7 且来自 cu128 索引；torchvision 会被索引自动配成对应版本
- 已验证的组合是 `torch 2.11.0+cu128` + `torchvision 0.26.0+cu128`
- 装完立刻自检：

```bat
envs\python.exe -c "import torch;print(torch.__version__, torch.version.cuda, torch.cuda.get_arch_list())"
:: 期望：2.11.0+cu128 / 12.8 / 列表里出现 'sm_120'
```

**同一份 cu128 也覆盖 20/30/40 系，不需要为不同卡分别打包。** 实测
`torch 2.11.0+cu128` 的 `torch.cuda.get_arch_list()` 为
`['sm_75','sm_80','sm_86','sm_90','sm_100','sm_120']`：

- 30 系（sm_86）、20 系（sm_75）有原生 cubin
- 40 系是 sm_89，**列表里没有**，但会走 sm_86 的 cubin——两者同属 8.x 大版本，
  NVIDIA 保证同大版本内次版本向上二进制兼容（现在的 cu121 构建同样没有 sm_89，4090 一直正常）
- 20 系**不因缺 bf16 而不可用**：`runtime_sam_features.py` 里有显式回退，
  `get_device_capability()[0] < 8` 时把 SAM-3D-Body backbone 从 bf16 转成 fp16（Turing 有 fp16 tensor core）；
  HaMeR 的设备门槛是 `CC >= 7.0 且显存 >= 4GB`；DINOv3 的 `layernormbf16` 只是 `nn.LayerNorm` 的配置标签，不强制 bf16。
  真正要留意的是**显存**（20 系多为 8–11 GB，preset 按 12 GiB 调）和**这条 fp16 分支未经验证**（我们的验证机是 3060，走的是 bf16 分支）。

## 3. 换 torch 之后必须重建的编译扩展

**一键流程用到的路径不需要重编**，两种构建都验证过。下面这些是**本机编译**的 `.pyd`，
链接了 torch 的 C++ ABI。只有在你自己再次更换 torch 版本、或者要跑依赖它们的训练路径时，
才需要处理，否则 import 会抛 undefined symbol。

| 文件 | 处理 |
|---|---|
| `third-party\detectron2-src\detectron2\_C.cp311-win_amd64.pyd` | **必须重编** |
| `third-party\mmcv-src-patched\mmcv\_ext.cp311-win_amd64.pyd` | 纯 CPU 扩展；注意它**不是**当前生效的那份 mmcv |
| `third-party\mmcv1-src-patched\mmcv\` | **当前生效**的 mmcv（可编辑安装指向这里），**没有**编译扩展，不用处理 |

重建 detectron2：

```bat
cd third-party\detectron2-src
..\..\envs\python.exe -m pip install --no-build-isolation -e .
```

- detectron2 是 0.6，非常老，对 torch 2.7+ 大概率需要打补丁。
  本包里它已经被 patch 过一轮，可参考 `tools\patch_*.py` 的思路。
- 成功判据：`envs\python.exe -c "import detectron2;print(detectron2.__version__)"`
- 两个扩展都是 **CPU-only**（二进制里没有任何 CUDA 符号），所以**不需要**配
  `TORCH_CUDA_ARCH_LIST`，也不用编 CUDA 内核。

关于可编辑安装的搬迁：`envs\Lib\site-packages\__editable___*_finder.py` 是按
`__file__` 相对定位的（`_PKG_ROOT` = 向上四级），所以整个包可以整体搬路径，
到新机器后**不需要**重新注册这些可编辑安装。涉及 detectron2 / mmcv / mmpose。

## 4. 代码层面的迁移风险（未全部验证）

- **`torch.load`**：全包 127 处，其中 **84 处没有写 `weights_only`**。
  torch >= 2.6 把默认值翻成 `True`，遇到含非张量对象的检查点会直接抛错。
  - **已实测安全**：`app\flowhmr\checkpoints\flowhmr_latest\flowhmr_latest.ckpt`
    （只含 `model_state_dict` + `epoch`/`global_step`，`weights_only=True` 能正常加载）
  - **未验证**：ViTPose / mmcv 的 checkpoint 工具
    （`third-party\ViTPose\mmcv_custom\checkpoint.py`、
    `third-party\mmcv1-src-patched\mmcv\runner\checkpoint.py`）
  - 报错时的修法：给对应调用显式加 `weights_only=False`
- **`torch.cuda.amp.autocast(...)` 带参数写法**：6 处，已弃用但仍可用。
  集中在 `app\flowhmr\flowhmr\core\motion\postprocess.py`。

## 5. 跑起来之后

包根有两个入口：

- `Video_to_FBX.bat "视频路径"` — 全链一键，拖视频上去即可
- `Open_Console.bat` — 本地网页控制台 `http://127.0.0.1:8760/`，
  可选视频、开始/停止、看实时进度、显存占用、过滤后的日志

环境自检：

```bat
envs\python.exe tools\check_env.py
```

权重自检（缺哪个会逐条列出，含获取来源）：

```
GET http://127.0.0.1:8760/api/preflight
```

权重清单：`weights_manifest.json`（28 个文件、18.036 GiB、带 SHA-256）。
一键流程实际必需 **13** 个文件，`console_server.py` 里的 `WEIGHTS` 列表是权威定义。

显存：

- 当前 preset 是按 **12 GiB** 调的，`max_generation_frames` 默认 3600（30fps 约 2 分钟）
- 5070 Ti 是 16 GB，可以调高，但**必须自己重测**显存预算，不要照搬 12GiB 的结论

## 6. 验证改动是否成功（必须做）

换 torch 后数值可能变化，不要假设一致。用固定素材跑完整流程，和下列基线对比：

| 项目 | cu121 构建（RTX 3060） | cu128 构建实测（RTX 3060） |
|---|---|---|
| 151 帧全链 | 约 260 秒 | **314.7 秒** |
| 565 帧全链 | 约 530 秒 | 未复测 |
| 相机阶段 torch 峰值（151 帧 / 16 张图） | — | 5.18 GiB |
| 生成阶段 torch 峰值（151 帧） | 约 7.25 GiB | 1.97 GiB（峰值按 151 帧计） |
| FBX 重导入 | 帧数一致、52 骨骼、1 蒙皮网格 | 同样通过 |

> 注意：在 Ampere 卡上 cu128 栈比原 cu121 栈慢约 19%（314.7 s vs 260 s），
> 这是换 torch 版本的代价，不影响正确性；50 系上本来就没得选。

判定要点：

1. `torch.cuda.get_arch_list()` 含 `sm_120`
2. `import detectron2` 成功
3. 一键流程跑通，`final.fbx` 能生成
4. 生成阶段数值与上表同量级；若差异明显，先查第 4 节的 `weights_only` 问题

## 7. 授权提醒（不要在公开场合转发权重）

`weights_manifest.json` 中部分权重**不可再分发**：

- SMPL-H / MANO 需要注册并接受许可，不可公开转载
- SAM-3D-Body、VGGT-Omega 是 gated 仓库，需要各自账号申请权限

这套代码本身可以传，权重按上面规则处理。

## 8. 包内结构速查

```
Video_to_FBX.bat        唯一一键入口
Open_Console.bat        网页控制台（127.0.0.1:8760）
pipeline_settings.json  全部默认参数
settings.bat            路径与环境变量（%PY% / %PACK_ROOT%）
weights_manifest.json   权重清单 + SHA-256
envs\                   自带 Python 3.11.4 + 依赖（torch 视构建而定，见第 0 节）
app\                    FlowHMR / HaMeR / SAM-3D-Body 等代码与权重
third-party\            detectron2 / mmcv / ViTPose 源码（可编辑安装目标）
tools\                  流水线脚本（video_to_fbx.py、console_server.py 等）
inputs\ outputs\        运行时创建（本包不附带示例视频与输出）
```
