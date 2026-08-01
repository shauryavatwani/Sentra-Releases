@echo off
REM ===========================================================================
REM  Sentra - Windows build script
REM
REM  Run this from the PROJECT ROOT on the Windows build machine:
REM      windows\build_windows.bat
REM
REM  Produces: windows\output\SentraSetup.exe
REM
REM  Set SENTRA_CONSOLE=1 first to build a version with a visible console
REM  window - do that for your first test build, because a windowed build that
REM  crashes during startup shows you nothing at all.
REM ===========================================================================

setlocal EnableDelayedExpansion
cd /d "%~dp0\.."
echo === Sentra Windows build ===
echo Project root: %CD%
echo.

REM --- Python version gate --------------------------------------------------
REM numpy is pinned to 1.26.4 (see requirements.txt for why it must stay <2).
REM That release has no wheels for Python 3.13+, so a newer Python fails with a
REM confusing "building wheel for numpy" compiler error instead of a clear one.
where py >nul 2>&1
if errorlevel 1 (
    echo ERROR: the Python launcher 'py' was not found.
    echo Install Python 3.12 from python.org and tick "Add Python to PATH".
    exit /b 1
)

set PYEXE=
for %%V in (3.12 3.11 3.10) do (
    if "!PYEXE!"=="" (
        py -%%V -c "import sys" >nul 2>&1
        if not errorlevel 1 set PYEXE=py -%%V
    )
)
if "%PYEXE%"=="" (
    echo ERROR: need Python 3.10, 3.11 or 3.12 ^(3.12 recommended^).
    echo Python 3.13+ will NOT work: numpy 1.26.4 has no wheels for it.
    py -0
    exit /b 1
)
echo Using interpreter: %PYEXE%
%PYEXE% -c "import sys; print('  ->', sys.version)"
echo.

REM --- Virtual environment --------------------------------------------------
if not exist ".venv-build\Scripts\python.exe" (
    echo [1/6] Creating build virtualenv...
    %PYEXE% -m venv .venv-build
    if errorlevel 1 exit /b 1
) else (
    echo [1/6] Reusing existing .venv-build
)
set VPY=.venv-build\Scripts\python.exe
%VPY% -m pip install --upgrade pip setuptools wheel --quiet
echo.

REM --- torch (CPU build) ----------------------------------------------------
REM Installed BEFORE requirements.txt and from PyTorch's cpu index. The default
REM PyPI torch on Windows is the CUDA build: ~2.5GB, all of which would end up
REM inside the installer for zero benefit, since the pose model runs on CPU.
echo [2/6] Installing torch (CPU-only build)...
%VPY% -m pip install torch==2.13.0 torchvision==0.28.0 ^
    --index-url https://download.pytorch.org/whl/cpu
if errorlevel 1 (
    echo ERROR: torch install failed.
    exit /b 1
)
echo.

echo [3/6] Installing application dependencies...
%VPY% -m pip install -r backend_v2\requirements.txt
if errorlevel 1 (
    echo ERROR: dependency install failed.
    exit /b 1
)
%VPY% -m pip install pyinstaller==6.11.1
if errorlevel 1 exit /b 1
echo.

REM --- InsightFace models ---------------------------------------------------
REM ~613MB, and the installer must ship them so the client PC needs no internet
REM on first run. Downloading here (on the build machine, which does have
REM internet) is what makes that possible.
echo [4/6] Checking InsightFace model set...
%VPY% -c "import pathlib,sys; p=pathlib.Path.home()/'.insightface'/'models'/'buffalo_l'; sys.exit(0 if p.is_dir() else 1)"
if errorlevel 1 (
    echo     Not cached yet - downloading buffalo_l ^(~613MB, one time^)...
    %VPY% -c "from insightface.app import FaceAnalysis; a=FaceAnalysis(providers=['CPUExecutionProvider']); a.prepare(ctx_id=0, det_size=(320,320)); print('  model set ready')"
    if errorlevel 1 (
        echo ERROR: could not download the InsightFace models.
        echo Copy the .insightface folder from the dev Mac into %%USERPROFILE%% instead.
        exit /b 1
    )
) else (
    echo     Already cached.
)
echo.

REM --- PyInstaller ----------------------------------------------------------
echo [5/6] Building the application bundle ^(this takes several minutes^)...
if exist "dist\Sentra" rmdir /s /q "dist\Sentra"
%VPY% -m PyInstaller windows\sentra.spec --noconfirm --clean
if errorlevel 1 (
    echo ERROR: PyInstaller build failed.
    exit /b 1
)
if not exist "dist\Sentra\Sentra.exe" (
    echo ERROR: dist\Sentra\Sentra.exe was not produced.
    exit /b 1
)
echo     Bundle: dist\Sentra\Sentra.exe
echo.

REM --- Inno Setup -----------------------------------------------------------
echo [6/6] Building the installer...
set ISCC=
for %%P in (
    "%ProgramFiles(x86)%\Inno Setup 6\ISCC.exe"
    "%ProgramFiles%\Inno Setup 6\ISCC.exe"
) do (
    if "!ISCC!"=="" if exist %%P set ISCC=%%P
)
if "%ISCC%"=="" (
    where iscc >nul 2>&1 && set ISCC=iscc
)
if "%ISCC%"=="" (
    echo.
    echo WARNING: Inno Setup 6 not found, so SentraSetup.exe was NOT built.
    echo The application bundle in dist\Sentra IS complete and runnable -
    echo only the installer wrapper is missing.
    echo Install from https://jrsoftware.org/isdl.php then re-run this script.
    exit /b 2
)
%ISCC% windows\sentra.iss
if errorlevel 1 (
    echo ERROR: Inno Setup compilation failed.
    exit /b 1
)
echo.

REM --- Release manifest -----------------------------------------------------
REM Generates the version.json that the in-app updater reads, with the real
REM sha256 of the installer that was just built. Doing it here is what keeps
REM the checksum honest - a hand-written manifest is a manifest that will
REM eventually be wrong, and a wrong checksum means every client refuses the
REM update with an integrity error.
echo [7/7] Generating the release manifest...
%VPY% windows\make_release.py
if errorlevel 1 (
    echo ERROR: could not generate version.json.
    exit /b 1
)

echo.
echo ============================================================
echo  BUILD COMPLETE
echo    Installer : windows\output\SentraSetup.exe
echo    Manifest  : windows\output\version.json
echo    Raw bundle: dist\Sentra\
echo.
echo  To publish an update, upload BOTH files to the same place
echo  and make sure the "url" in version.json points at the
echo  installer. See windows\output\PUBLISHING.txt
echo ============================================================
endlocal
