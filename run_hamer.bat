@echo off
rem HaMeR: default N=2, interpolate output rotations, fixed TTA=1
rem Usage: run_hamer.bat "video" "main_track.npz" "out_dir" [--detect_interval N]
call "%~dp0settings.bat"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0tools\run_hamer_pose.ps1" %*
exit /b %ERRORLEVEL%
