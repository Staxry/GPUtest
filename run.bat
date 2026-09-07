@echo off
chcp 65001 >nul
cd /d "%~dp0"

where python >nul 2>&1
if %errorlevel% NEQ 0 (
  echo Python не найден. Запускаю проверку зависимостей...
  call setup.bat
  if errorlevel 1 exit /b 1
)

python -c "from coreforge.deps import required_ok; raise SystemExit(0 if required_ok() else 1)" 2>nul
if %errorlevel% NEQ 0 (
  echo Не хватает драйвера или Python. Проверяю зависимости...
  echo.
  call setup.bat
  if errorlevel 1 exit /b 1
)

where pythonw >nul 2>&1
if %errorlevel%==0 (
  start "" pythonw main.py
  exit /b 0
)
pythonw main.py 2>nul
if %errorlevel%==0 exit /b 0
python main.py
if errorlevel 1 pause
