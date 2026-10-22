# Rebuilding the cu128 package (RTX 50-series)

`build_cu128_package.ps1` turns a working **cu121** pack into the **cu128** delivery
package that runs on Blackwell cards (RTX 50-series, `sm_120`).

## Why this exists

`sm_120` needs the cu128 wheels. The same wheels still run on Turing / Ampere / Ada,
but cu128 is measurably slower on those cards — on an RTX 3060 the full 151-frame
chain takes **314.7 s** with cu128 vs **260 s** with cu121 (~19% slower). So the
working pack stays on cu121 and cu128 is only installed for the delivery build.

This folder therefore holds the **only** cu128 artefacts you need to keep:
the two wheels (2.6 GB). The built package is ~19 GB and can always be regenerated.

## What you need

1. **The two wheels**, in the same folder as the script:

   ```powershell
   pip download torch==2.11.0+cu128 torchvision==0.26.0+cu128 `
       --index-url https://download.pytorch.org/whl/cu128 --no-deps -d .
   ```

   Expected files:
   - `torch-2.11.0+cu128-cp311-cp311-win_amd64.whl` (2.56 GB)
   - `torchvision-0.26.0+cu128-cp311-cp311-win_amd64.whl` (8.9 MB)

   The wheels are **not** committed to this repository — they are far too large.
   Any Python 3.11 / Windows x64 machine can download them; the pack's own
   `envs\python.exe` works too.

2. **7-Zip.** The script looks for `D:\7-Zip\7z.exe`, the two standard
   `Program Files` locations, then `7z.exe` on `PATH`. Override with `-SevenZip`.

3. **~70 GB free** on the drive that holds the staging directory
   (pack 23 GB + staging 24 GB + output 19 GB).

## Usage

```powershell
powershell -ExecutionPolicy Bypass -File build_cu128_package.ps1
```

| Switch | Effect |
|---|---|
| `-DryRun` | resolve and print everything, create nothing |
| `-SkipZip` | prepare and verify the staging env, then delete it (no archive) |
| `-KeepStage` | keep the prepared staging directory for inspection |

| Parameter | Default | Notes |
|---|---|---|
| `-PackRoot` | auto-detected | directory containing `envs\python.exe`; searched: parent of this folder, its sibling `flowhmr_pack`, `F:\flowhmr_pack` |
| `-StageDir` | `<pack parent>\_build_cu128` | temporary copy |
| `-OutDir` | `<pack parent>\flowhmr_share` | where the volumes are written |
| `-ZipName` | `FlowHMR_kit_50series.zip` | archive base name |
| `-SevenZip` | auto-detected | full path to `7z.exe` |

## What the script does

1. Copies the pack to the staging directory with **robocopy**, excluding
   `logs\`, `inputs\`, `outputs\`, all `*.mp4/*.mkv/*.mov/*.avi/*.webm/*.llc`
   and `__pycache__`.
2. Installs the two wheels into the staged environment with
   `pip install --no-deps --force-reinstall`.
3. Verifies the staged environment:
   - `torch.__version__` must end with `+cu128`
   - `torch.cuda.get_arch_list()` must contain **`sm_120`** (50-series)
     and **`sm_86`** (Ampere)
4. Archives the staging directory into 4 GiB volumes:
   `FlowHMR_kit_50series.zip.001 … .00N`.
5. Deletes the staging directory.

**The working pack is never modified** — everything happens in the staging copy.

## Expected output

```
  torch 2.11.0+cu128 torchvision 0.26.0+cu128
  cuda 12.8
  arch ['sm_75', 'sm_80', 'sm_86', 'sm_90', 'sm_100', 'sm_120']
  OK
```

Measured reference (RTX 3060 12 GB, 151-frame clip, full chain):

| Stack | Chain time | Notes |
|---|---|---|
| cu121 (torch 2.5.1) | ~260 s | local default, faster on Ampere |
| cu128 (torch 2.11.0) | 314.7 s | required for 50-series |

`sm_89` (RTX 40-series) is absent from both arch lists; those cards run the
`sm_86` cubin, which is binary-compatible within the same major version.

## Refreshing the worked example

If the pack changes, just re-run the script — no version numbers live in it
except the wheel filenames it looks for (`torch-*+cu128-*.whl`).
