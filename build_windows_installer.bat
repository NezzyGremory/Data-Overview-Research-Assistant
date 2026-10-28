@echo off
setlocal
cd /d "%~dp0"

set "PYTHON=.venv\Scripts\python.exe"
set "APP_NAME=DORA-v7.0.0"
set "APP_EXE_NAME=DORA.exe"
set "APP_VERSION=7.0.0"
set "BUILD_DIR=build\installer-dora-v7.0.0"
set "PAYLOAD_ROOT=hasil_compile\%APP_NAME%\payload"
set "PAYLOAD_DIR=%PAYLOAD_ROOT%\DORA"
set "INSTALLER=hasil_compile\DORA-v7.0.0-Setup.exe"
set "ISCC="

if not exist "%PYTHON%" (
    echo [ERROR] Project virtual environment not found: %PYTHON%
    echo Create it and install requirements.txt before building.
    exit /b 1
)

if not exist "installer\DatasetResearch.iss" (
    echo [ERROR] Installer definition not found: installer\DatasetResearch.iss
    exit /b 1
)

echo [0/3] Preparing an incremental build for v%APP_VERSION%...
if exist "%PAYLOAD_DIR%" rmdir /s /q "%PAYLOAD_DIR%"
if exist "%INSTALLER%" del /q "%INSTALLER%"
if not exist "%BUILD_DIR%" mkdir "%BUILD_DIR%"
if not exist "%PAYLOAD_ROOT%" mkdir "%PAYLOAD_ROOT%"

echo [1/3] Building the optimized one-directory application...
"%PYTHON%" -m PyInstaller --noconfirm --distpath "%CD%\%PAYLOAD_ROOT%" --workpath "%CD%\%BUILD_DIR%" "Dataset Research.spec"
if errorlevel 1 (
    echo [ERROR] PyInstaller build failed.
    exit /b 1
)

if defined SIGN_CERT_SHA1 (
    where signtool.exe >nul 2>nul
    if errorlevel 1 (
        echo [ERROR] signtool.exe is not available. The application remains unsigned.
        exit /b 1
    )
    signtool.exe sign /sha1 "%SIGN_CERT_SHA1%" /fd SHA256 /tr "http://timestamp.digicert.com" /td SHA256 "%PAYLOAD_DIR%\%APP_EXE_NAME%"
    if errorlevel 1 exit /b 1
    signtool.exe verify /pa /v "%PAYLOAD_DIR%\%APP_EXE_NAME%"
    if errorlevel 1 exit /b 1
)

echo [2/3] Locating Inno Setup compiler...
where ISCC.exe >nul 2>nul
if not errorlevel 1 set "ISCC=ISCC.exe"
if not defined ISCC if exist "%LOCALAPPDATA%\Programs\Inno Setup 6\ISCC.exe" set "ISCC=%LOCALAPPDATA%\Programs\Inno Setup 6\ISCC.exe"
if not defined ISCC if exist "%ProgramFiles(x86)%\Inno Setup 6\ISCC.exe" set "ISCC=%ProgramFiles(x86)%\Inno Setup 6\ISCC.exe"
if not defined ISCC if exist "%ProgramFiles%\Inno Setup 6\ISCC.exe" set "ISCC=%ProgramFiles%\Inno Setup 6\ISCC.exe"

if not defined ISCC (
    echo [ERROR] Inno Setup 6 compiler ISCC.exe was not found.
    echo The application files are available in %PAYLOAD_DIR%\
    exit /b 1
)

echo [3/3] Compiling the installer with Inno Setup...
"%ISCC%" "installer\DatasetResearch.iss"
if errorlevel 1 (
    echo [ERROR] Inno Setup compilation failed.
    exit /b 1
)

echo.
echo SUCCESS: %INSTALLER%

if defined SIGN_CERT_SHA1 (
    echo [INFO] SIGN_CERT_SHA1 is set. Sign the installer with the configured code-signing certificate after packaging.
    where signtool.exe >nul 2>nul
    if errorlevel 1 (
        echo [ERROR] signtool.exe is not available. The installer remains unsigned.
        exit /b 1
    )
    signtool.exe sign /sha1 "%SIGN_CERT_SHA1%" /fd SHA256 /tr "http://timestamp.digicert.com" /td SHA256 "%INSTALLER%"
    if errorlevel 1 (
        echo [ERROR] Code signing failed. Check the certificate, private key access, and timestamp-server connection.
        exit /b 1
    )
    signtool.exe verify /pa /v "%INSTALLER%"
    if errorlevel 1 exit /b 1
) else (
    echo [NOTE] No SIGN_CERT_SHA1 was supplied. Windows may show SmartScreen for this unsigned installer.
)

exit /b 0
