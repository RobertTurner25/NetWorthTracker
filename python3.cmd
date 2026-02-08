@echo off
setlocal

set "SCRIPT_DIR=%~dp0"
set "VENV_PYTHON=%SCRIPT_DIR%.venv\Scripts\python.exe"

if exist "%VENV_PYTHON%" (
  "%VENV_PYTHON%" %*
  exit /b %ERRORLEVEL%
)

where python >nul 2>&1
if %ERRORLEVEL%==0 (
  python %*
  exit /b %ERRORLEVEL%
)

echo python3 shim could not find a Python interpreter.
echo Expected: %VENV_PYTHON%
echo Or install Python for Windows and ensure `python` is on PATH.
exit /b 1
