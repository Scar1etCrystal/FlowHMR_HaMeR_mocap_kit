@echo off
rem ============================================================
rem  INTERNAL DIAGNOSTIC ENTRY -- NOT the normal entry point.
rem
rem  Normal use: drag a video onto Video_to_FBX.bat in the package root.
rem  This script only runs the FlowHMR stage by itself, without SAM hand
rem  features, HaMeR fingers or the merge step, so its output is NOT the
rem  finished deliverable.
rem
rem  Usage: run_flowhmr.bat <video> [extra args...]
rem ============================================================
call "%~dp0..\..\settings.bat"
cd /d "%FLOWHMR%"
if not exist "%LOGS%" mkdir "%LOGS%"
echo [FlowHaMeR][internal] input: %*
"%PY%" run_v2m_demo_generation.py %*
exit /b %ERRORLEVEL%
