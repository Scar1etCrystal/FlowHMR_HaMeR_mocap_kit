@echo off
chcp 65001 >nul
setlocal
call "%~dp0settings.bat"
echo Starting the local console: http://127.0.0.1:8760/
echo The browser opens automatically. Close this window to stop the server.
echo.
"%PY%" -u "%PACK_ROOT%\tools\console_server.py" %*
echo.
echo Server stopped.
if not defined FLOWHMR_NO_PAUSE pause
