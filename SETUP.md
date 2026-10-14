# Setup on a new PC (Windows + VSCode, no Cursor required)

This is the short, copy-paste guide for cloning this repository onto a fresh
Windows machine and getting every notebook and test running locally.
**Cursor is not required** — VSCode is fully sufficient.

---

## 1. Prerequisites (install once per machine)

Install these from their official installers — accept the defaults unless noted.

| Tool | Why | Notes |
|------|-----|-------|
| **Python 3.10 – 3.13** | Runs the `tdse` package, notebooks, tests | <https://www.python.org/downloads/windows/> — **tick "Add Python to PATH"** during install. The `py` launcher that ships with the installer is what `setup_env.bat` looks for first. `pyproject.toml` requires `>=3.10`; the build is regularly tested on 3.10/3.11/3.12 in CI and works on 3.13 locally. |
| **Git for Windows** | Clone / pull / push the repo | <https://git-scm.com/download/win> |
| **VSCode** | Editor + notebook host | <https://code.visualstudio.com/> |
| **VSCode extensions** | Python + notebook support | Inside VSCode, install: `ms-python.python`, `ms-toolsai.jupyter`. (Optional but nice: `ms-python.vscode-pylance`.) |

Verify the basics from a fresh PowerShell window:

```powershell
py -3 --version       # should print Python 3.10.x / 3.11.x / 3.12.x / 3.13.x
git --version
code --version
```

---

## 2. Clone the repository

Pick a folder you control (avoid OneDrive / Dropbox / other syncing folders if
possible — they occasionally lock files inside `.venv` while pip is writing to
them). Then:

```powershell
cd %USERPROFILE%\source\repos        # or wherever you keep code
git clone https://github.com/Alex-G-Gauthier/tdse-leapfrog.git
cd tdse-leapfrog
```

> The `pyproject.toml`, `requirements.txt`, `tdse\`, `notebooks\`, etc., live at
> the root of the cloned repo. That is where you run `setup_env.bat` from.

---

## 3. Run the one-shot environment setup

From the repo root (the folder that contains `pyproject.toml`):

```powershell
setup_env.bat
```

What it does, in order:

1. Finds a Python ≥ 3.10 via the `py` launcher (falls back to `python` on PATH).
2. Creates a local virtual environment in `.venv\` (gitignored, never pushed).
3. Upgrades `pip`, `setuptools`, `wheel`.
4. Editable-installs the `tdse` package with the `notebooks` and `test` extras
   — i.e. NumPy, SciPy, Matplotlib, imageio (+ ffmpeg), Pillow, tqdm, Jupyter,
   ipykernel, nbformat, nbconvert, pytest, pytest-timeout.
5. Registers a Jupyter kernel named **"Python (tdse)"** so VSCode and JupyterLab
   can pick it.
6. Runs the fast pytest tier (`pytest -m "not slow"`, ~2 s) as a smoke check.

The script is idempotent — re-run it any time. To start clean, delete `.venv\`
and run it again.

If you would rather do it by hand, the equivalent commands are:

```powershell
py -3.12 -m venv .venv                       REM or -3.13 / -3.11 / -3.10
.venv\Scripts\activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e ".[notebooks,test]"
python -m ipykernel install --user --name tdse --display-name "Python (tdse)"
pytest -m "not slow"
```

---

## 4. Use the project from VSCode

1. `code .` from the repo root (or **File → Open Folder…** and pick the cloned
   folder).
2. When VSCode prompts for a Python interpreter, choose
   `.\.venv\Scripts\python.exe` (or run **Ctrl+Shift+P → Python: Select
   Interpreter**).
3. Open any notebook under `notebooks\` and, in the top-right kernel picker,
   select **"Python (tdse)"** (or the `.venv` interpreter directly).
4. Run cells with **Shift+Enter**. The first run of a heavy notebook (notably
   `03_time_dependent.ipynb`) takes a few minutes.

---

## 5. Common workflows

```powershell
.venv\Scripts\activate                       REM activate the venv in this shell

REM --- run the test suites -----------------------------------------------------
pytest -m "not slow"                         REM ~2 s,  fast tier
pytest -m slow                               REM ~10 s, §3-§9 physics tier
pytest                                       REM full suite

REM --- regenerate every notebook end-to-end ------------------------------------
jupyter nbconvert --to notebook --execute --inplace `
    notebooks\00_theory_and_method.ipynb `
    notebooks\01_free_and_bound_states.ipynb `
    notebooks\02_scattering.ipynb `
    notebooks\03_time_dependent.ipynb `
    notebooks\04_breaking_points.ipynb

REM --- regenerate the read-only walk-through notebook --------------------------
python tools\build_final_notebook.py         REM rewrites "Official Final Notebook.ipynb" at repo root
```

---

## 6. Pulling later updates

```powershell
cd path\to\Alex
git pull
setup_env.bat                                REM safe to re-run; picks up new deps
```

---

## 7. Troubleshooting

| Symptom | Fix |
|---------|-----|
| `setup_env.bat` says no Python ≥ 3.10 found | Install Python from python.org and tick "Add Python to PATH". Open a **new** PowerShell window so PATH refreshes. |
| `pip install -e .` is extremely slow or fails with file-lock errors | The repo is sitting inside OneDrive / Dropbox / a syncing folder. Move it to a non-synced path (e.g. `C:\src\Alex`). |
| VSCode does not show the `tdse` kernel | Run **Ctrl+Shift+P → Jupyter: Select Interpreter to Start Jupyter Server** and pick `.venv\Scripts\python.exe`, or rerun `python -m ipykernel install --user --name tdse --display-name "Python (tdse)"` from inside the activated venv. |
| `imageio` complains about ffmpeg when writing GIFs | `imageio-ffmpeg` is in the dependency list and bundles a binary; reinstall with `pip install --force-reinstall imageio-ffmpeg`. |
| Notebook math renders but a few glyphs (e.g. `\mathcal`) look wrong | Matplotlib's default `mathtext.fontset` (`dejavusans`) is missing some math glyphs; switch to `cm` with `mpl.rcParams['mathtext.fontset'] = 'cm'` at the top of the notebook. |
