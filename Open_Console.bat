@echo off
chcp 65001 >nul
setlocal
call "%~dp0settings.bat"
echo ============================================================
echo  MOCAP CONSOLE
echo  http://127.0.0.1:8760/
echo ============================================================
echo.
echo Starting... if no browser tab opens, paste the address above.
echo.
"%PY%" -u "%PACK_ROOT%\tools\console_server.py" %*
set "CODE=%ERRORLEVEL%"
echo.
echo ------------------------------------------------------------
if "%CODE%"=="0" (
  echo The console is running at http://127.0.0.1:8760/
  echo Close that browser tab when done; this server stops itself
  echo about 25 seconds later. You can also close this window now.
) else (
  echo The console FAILED to start. Exit code %CODE%.
  echo Details are in this window and in logs\console_server_last.log
)
echo ------------------------------------------------------------
if not defined FLOWHMR_NO_PAUSE pause
exit /b %CODE%
