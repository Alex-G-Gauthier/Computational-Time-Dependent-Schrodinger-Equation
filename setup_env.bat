@echo off
REM ============================================================================
REM  setup_env.bat  --  one-shot local Python environment setup for the
REM                    1D TDSE leapfrog project (tdse package).
REM
REM  What it does:
REM    1. Locates a Python >= 3.10 interpreter (py launcher preferred).
REM    2. Creates a local virtual environment in .venv\
REM    3. Upgrades pip / setuptools / wheel.
REM    4. Editable-installs the tdse package with [notebooks,test] extras.
REM    5. Registers a Jupyter kernel named "tdse" pointing at the venv.
REM    6. Runs the fast pytest tier as a smoke check.
REM
REM  Usage (from the repo root, the folder containing pyproject.toml):
REM      setup_env.bat
REM
REM  Re-run any time -- it is idempotent.  Delete .venv\ to start clean.
REM ============================================================================

setlocal enabledelayedexpansion
cd /d "%~dp0"

echo.
echo ============================================================
echo  TDSE project -- local environment setup
echo  Working directory: %CD%
echo ============================================================
echo.

REM --- 0. Sanity check: are we in the right folder? ---------------------------
if not exist "pyproject.toml" (
    echo [ERROR] pyproject.toml not found in %CD%.
    echo         Run this script from the repo root ^(the folder containing
    echo         pyproject.toml, requirements.txt, and the tdse\ package^).
    goto :fail
)

REM --- 1. Find a usable Python interpreter (>= 3.10) --------------------------
set "PYEXE="

where py >nul 2>nul
if !errorlevel! == 0 (
    for %%V in (3.13 3.12 3.11 3.10) do (
        py -%%V -c "import sys; sys.exit(0)" >nul 2>nul
        if !errorlevel! == 0 (
            set "PYEXE=py -%%V"
            echo [OK] Found Python %%V via the py launcher.
            goto :have_python
        )
    )
)

where python >nul 2>nul
if !errorlevel! == 0 (
    for /f "tokens=2 delims= " %%V in ('python --version 2^>^&1') do set "PYVER=%%V"
    echo [INFO] python.exe is version !PYVER!
    python -c "import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)" >nul 2>nul
    if !errorlevel! == 0 (
        set "PYEXE=python"
        echo [OK] Using python on PATH.
        goto :have_python
    )
)

echo [ERROR] No Python ^>= 3.10 was found on this machine.
echo         Install Python 3.11 or 3.12 from https://www.python.org/downloads/
echo         and tick "Add Python to PATH" during installation, then re-run.
goto :fail

:have_python

REM --- 2. Create the virtual environment --------------------------------------
if exist ".venv\Scripts\python.exe" (
    echo [OK] .venv already exists -- reusing it.
) else (
    echo [..] Creating virtual environment in .venv\
    %PYEXE% -m venv .venv
    if errorlevel 1 (
        echo [ERROR] venv creation failed.
        goto :fail
    )
)

set "VENV_PY=.venv\Scripts\python.exe"
if not exist "%VENV_PY%" (
    echo [ERROR] %VENV_PY% missing after venv creation.
    goto :fail
)

REM --- 3. Upgrade pip toolchain -----------------------------------------------
echo.
echo [..] Upgrading pip / setuptools / wheel inside the venv...
"%VENV_PY%" -m pip install --upgrade pip setuptools wheel
if errorlevel 1 goto :fail

REM --- 4. Editable install with notebook + test extras ------------------------
echo.
echo [..] Installing the tdse package (editable) with [notebooks,test] extras...
"%VENV_PY%" -m pip install -e ".[notebooks,test]"
if errorlevel 1 goto :fail

REM --- 5. Register a Jupyter kernel named "tdse" ------------------------------
echo.
echo [..] Registering Jupyter kernel "tdse"...
"%VENV_PY%" -m ipykernel install --user --name tdse --display-name "Python (tdse)"
if errorlevel 1 (
    echo [WARN] ipykernel registration failed -- you can still run notebooks
    echo        from VSCode by selecting .venv\Scripts\python.exe directly.
)

REM --- 6. Smoke test: fast pytest tier ----------------------------------------
echo.
echo ============================================================
echo  Running fast test tier as a smoke check (pytest -m "not slow")
echo ============================================================
"%VENV_PY%" -m pytest -m "not slow" -q
set "PYTEST_RC=!errorlevel!"

echo.
echo ============================================================
if !PYTEST_RC! == 0 (
    echo  SUCCESS -- environment ready.
) else (
    echo  Setup completed but the fast tests returned a non-zero status.
    echo  Inspect the output above; the venv itself is still usable.
)
echo.
echo  Next steps:
echo    1. Activate the venv in this shell:   .venv\Scripts\activate
echo    2. Launch Jupyter Lab:                jupyter lab
echo       ...or open the repo folder in VSCode and pick the
echo       "Python (tdse)" kernel for any notebook in notebooks\.
echo    3. Full test suite:                   pytest
echo ============================================================
echo.
endlocal
exit /b %PYTEST_RC%

:fail
echo.
echo [ABORTED] Setup did not complete.  See messages above.
endlocal
exit /b 1
