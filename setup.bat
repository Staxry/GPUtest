@echo off
chcp 65001 >nul
cd /d "%~dp0"
title CoreForge — проверка зависимостей
echo.

where python >nul 2>&1
if %errorlevel%==0 goto :have_python

echo Python не найден в PATH.
where winget >nul 2>&1
if %errorlevel%==0 (
  echo Ставлю Python 3.12 через winget...
  winget install -e --id Python.Python.3.12 --accept-package-agreements --accept-source-agreements --disable-interactivity
  echo.
  echo Если Python только что поставился — закройте это окно и запустите setup.bat снова.
  echo Ссылка, если winget не сработал: https://www.python.org/downloads/windows/
  pause
  exit /b 1
)

echo winget тоже нет. Скачайте Python 3.10+ вручную:
echo https://www.python.org/downloads/windows/
echo При установке отметьте "Add python.exe to PATH".
pause
exit /b 1

:have_python
python -m coreforge.deps
set ERR=%errorlevel%
echo.
if %ERR% NEQ 0 (
  echo Драйвер NVIDIA: https://www.nvidia.com/Download/index.aspx?lang=ru
  echo Драйвер AMD:    https://www.amd.com/en/support/download/drivers.html
  echo.
)
pause
exit /b %ERR%
