@echo off
rem ============================================================
rem  INTERNAL DIAGNOSTIC ENTRY -- NOT the normal entry point.
rem
rem  Normal use: drag a video onto Video_to_FBX.bat in the package root.
rem  This script only runs the HaMeR stage by itself; it needs an existing
rem  main_track.npz and its output still has to be merged into a body result.
rem
rem  Usage: run_hamer.bat "video" "main_track.npz" "out_dir" [--detect_interval N]
rem ============================================================
call "%~dp0..\..\settings.bat"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0..\run_hamer_pose.ps1" %*
exit /b %ERRORLEVEL%
