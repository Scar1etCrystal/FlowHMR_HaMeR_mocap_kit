# Weights and environment

This repository contains source code only. Model weights and the Python
environment are intentionally **not** committed.

## Why not in Git

The weights this kit loads add up to **18.04 GiB across 28 files**. They are
third-party artifacts with their own terms, and two of them cannot legally be
mirrored at all:

- SMPL-H / MANO body models require registration and licence acceptance. They
  are not redistributable.
- SAM-3D-Body and VGGT-Omega are distributed from gated Hugging Face
  repositories, so a download requires your own account and an access request.

A Git repository is also the wrong transport for them: history would keep every
future revision forever, and GitHub rejects files above 100 MiB.

`weights_manifest.json` lists every file this kit expects, with its exact size
and SHA-256, so you can confirm an assembled set matches the one this code was
validated against.

## Where each file comes from

| Target path | Source |
|---|---|
| `checkpoints/<name>/<name>.ckpt` | FlowHMR base / latest release |
| `ckpts/yolox/yolox_l.pth` | `https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_l.pth` |
| `ckpts/vggt-omega/vggt_omega_1b_512.pt` | `huggingface-cli download facebook/VGGT-Omega` (gated) |
| `ckpts/sam-3d-body-dinov3/` | `huggingface-cli download facebook/sam-3d-body-dinov3` (gated) |
| `assets/body_models/smplh/neutral/model.npz` | AMASS `smplh_amass_neutral` release (MANO registration required) |
| `assets/body_models/smpl_neutral_J_regressor.pt` | GVHMR public asset |
| `app/hamer/_DATA/**` | HaMeR checkpoint and ViTPose whole-body weights from their official releases |
| `app/hamer/models/**` | Local HaMeR / body-model copies used by this package |

The upstream instructions in `app/flowhmr/docs/install.md` and the layout table
in `app/flowhmr/docs/resources.md` are the authoritative acquisition steps; this
file only records which files the packaged entry points actually read.

## Recommended handling

1. **Keep weights out of Git.** This repository ships code plus the manifest.
2. **Assemble them next to the code** using the upstream install commands, then
   verify against `weights_manifest.json`.
3. **If you want a ready-to-run download**, publish the assembled tree as a
   separate artifact rather than a commit: split it into under-2-GiB parts and
   attach them to a GitHub Release, or host it on Hugging Face / cloud storage.
   Do not re-upload SMPL-H/MANO or the gated FAIR checkpoints publicly.

## Environment

The runtime this project was validated on:

- Windows, Python **3.11.4**
- PyTorch **2.5.1+cu121**, CUDA 12.1 runtime, cuDNN 9.1.0
- 12 GiB GPU is the design target

`env/requirements-lock-win-py311.txt` is a full `pip freeze` of that
environment with local machine paths removed. Three packages are editable
installs that must be built from this repository, and they have to be installed
before the rest of the lock for the same versions to resolve:

```
pip install --no-build-isolation --no-deps -e third-party/detectron2-src
pip install --no-build-isolation --no-deps -e third-party/mmcv1-src-patched
pip install --no-build-isolation --no-deps -e third-party/ViTPose
pip install -r env/requirements-lock-win-py311.txt
```

`fbxsdkpy` (FBX export) is not on PyPI; see the header of the lock file for its
index URL.

Notes:

- The bundled environment directory is 5.16 GiB. Rebuild it from the lock
  instead of shipping it; compiled wheels are machine-specific.
- The packaged `settings.bat` assumes an interpreter at `envs\python.exe`
  relative to the package root. Point it at your own environment or keep that
  layout.
- `_DATA`, `models`, `assets`, `checkpoints`, `ckpts` and `envs` are covered by
  `.gitignore`, so assembling locally will not accidentally stage them.
