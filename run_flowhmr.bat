@echo off
rem ============================================================
rem  FlowHMR main pipeline: video -> SMPL-H motion (npz + fbx)
rem  Usage: run_flowhmr.bat <video> [extra args...]
rem  Example:
rem    run_flowhmr.bat "..\inputs\example.mp4" --vggt_interval 10 --postprocess
rem ============================================================
call "%~dp0settings.bat"
cd /d "%FLOWHMR%"
if not exist "%LOGS%" mkdir "%LOGS%"
echo [FlowHaMeR] input: %*
"%PY%" run_v2m_demo_generation.py %*
exit /b %ERRORLEVEL%
