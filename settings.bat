@echo off
rem FlowHaMeR pack - paths and verified runtime settings
set "PACK_ROOT=%~dp0"
if "%PACK_ROOT:~-1%"=="\" set "PACK_ROOT=%PACK_ROOT:~0,-1%"
set "APP=%PACK_ROOT%\app"
set "FLOWHMR=%APP%\flowhmr"
set "HAMER=%APP%\hamer"
set "PY=%PACK_ROOT%\envs\python.exe"
set "OUT=%PACK_ROOT%\outputs"
set "LOGS=%PACK_ROOT%\logs"
rem Fast-SAM: near-equal numerical results; cosine similarity is NOT bit equality.
set "SKIP_KEYPOINT_PROMPT=1"
set "IMG_SIZE=512"
set "LAYER_DTYPE=fp32"
set "GPU_HAND_PREP=1"
set "MHR_USE_CUDA_GRAPH=0"
rem Stage residency: park other models BEFORE VGGT; defer SAM; restore generation later.
rem Budgets below are tuned for a 12GiB card; re-measure before raising them.
set "FLOWHMR_UNLOAD_BETWEEN_STAGES=1"
rem Skip ONLY FlowHMR's ordinary 3-frame filter; other temporal constraints remain.
set "POSTPROCESS_SKIP_SMOOTH=1"
set "DINOV3_LOCAL_REPO=%FLOWHMR%\third_party\dinov3_hub\facebookresearch_dinov3_main"
set "HAMER_THREADS=8"
set "OMP_NUM_THREADS=8"
set "MKL_NUM_THREADS=8"
set "OPENBLAS_NUM_THREADS=8"
set "NUMEXPR_NUM_THREADS=8"
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
set "PYTHONPATH=%FLOWHMR%;%FLOWHMR%\third_party\sam-3d-body;%FLOWHMR%\third_party\vggt-omega;%FLOWHMR%\third_party\YOLOX"
