@echo off
chcp 65001 >nul
setlocal
call "%~dp0settings.bat"
if "%~1"=="" (
  echo Drop one or more video files onto this BAT.
  echo Output: %PACK_ROOT%\outputs\video_to_fbx\VIDEO_ID\final.fbx
  if not defined FLOWHMR_NO_PAUSE pause
  exit /b 2
)
echo One-click video to merged FBX.  Settings: pipeline_settings.json
echo SAM hands every 4 frames + linear interpolation, HaMeR interval 2
echo FlowHMR postprocess ON, ordinary 3-frame smoothing OFF
echo.
"%PY%" -u "%PACK_ROOT%\tools\video_to_fbx.py" %*
set "PIPELINE_EXIT=%ERRORLEVEL%"
echo.
if "%PIPELINE_EXIT%"=="0" (
  echo Finished. See the DONE path above for final.fbx.
) else (
  echo Failed. Read the stage log under outputs\video_to_fbx.
)
if not defined FLOWHMR_NO_PAUSE pause
exit /b %PIPELINE_EXIT%
