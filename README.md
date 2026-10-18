# PAMPI Notebooks

Jupyter notebooks for the lecture _Practical Parallel Algorithms with MPI_ (PAMPI), winter term 2026/27.
The C code cells in the notebooks are compiled with `mpicc` and run with `mpirun`, directly from JupyterLab.

Start with [`overview.ipynb`](overview.ipynb).

## Getting the Material and Updates

The notebooks are published lecture by lecture in a git repository.
Clone it once, then set up and start JupyterLab as described below:

```bash
git clone <REPOSITORY-URL> pampi-notebook
cd pampi-notebook
make
```

After each lecture, fetch the new notebooks from within this folder:

```bash
make update
```

`make update` keeps your own changes to the notebooks: new notebooks are added, and where an update and one of your edits touch the same lines, your version wins.
To get the original version of a notebook back, delete it and run `make update` again.

## Requirements

* an MPI implementation that provides `mpicc` and `mpirun` (Open MPI, MPICH, Intel MPI, …)
* a C compiler (used by `mpicc`)
* Python 3.9 or newer

Check that MPI works before setting up Jupyter:

```bash
mpicc --version
mpirun -n 2 hostname
```

On macOS, `brew install open-mpi` installs MPI, and `brew install libomp` the OpenMP runtime needed by the hybrid MPI+OpenMP notebook; on Debian/Ubuntu, `sudo apt install libopenmpi-dev openmpi-bin`.
On a cluster, load the MPI module first, e.g. `module load intelmpi`.

## Setup

The notebooks need a Python environment containing JupyterLab (or at least `ipykernel`) and `matplotlib`.
Install it into a dedicated virtual environment: unlike a package manager's JupyterLab installation, it is not replaced by system updates, so the packages stay installed.

The quickest way is the Makefile in the root of this repository:

```bash
make init    # create the environment .venv and install the packages from requirements.txt
make         # start JupyterLab (runs init first if needed)
make clean   # clear all notebook outputs, remove checkpoints, caches and material/data/
make check   # check that MPI, OpenMP and the environment work
make test    # run all notebooks headless and report failing cells (takes a few minutes)
make update  # fetch new notebooks from the course repository, keep your changes
```

`make test` writes the executed notebooks to `.test/`, the originals stay unchanged; `make test NOTEBOOKS=material/mpi-io.ipynb` runs a single notebook.
Since the `%%mpi` magic reports errors as `[ICE] …` messages instead of exceptions, the test looks for these messages.
Cells that fail on purpose (deadlock and error demos, exercise templates) carry the cell tag `expect-ice-error` and fail the test if they *succeed*.

`make VENV=~/.venvs/pampi` uses a different location for the environment.
The Makefile uses uv if it is installed, and `python3 -m venv` with pip otherwise.
To set up the environment by hand, follow one of the options below.

The commands below use [uv](https://docs.astral.sh/uv/).
Without uv, replace `uv venv DIR` with `python3 -m venv DIR` and `uv pip install --python DIR ...` with `DIR/bin/pip install ...`.

### Option A: Everything in One Environment (Recommended)

Install JupyterLab together with all packages the notebooks need:

```bash
uv venv ~/.venvs/pampi
uv pip install --python ~/.venvs/pampi jupyterlab matplotlib
```

Start JupyterLab from the root of this repository:

```bash
cd path/to/pampi-notebook
~/.venvs/pampi/bin/jupyter lab
```

The default kernel `Python 3 (ipykernel)` now runs in `~/.venvs/pampi`, so the notebooks work without any further configuration.
Optionally, add an alias to your shell configuration (e.g. `~/.zshrc`):

```bash
alias pampi-lab='~/.venvs/pampi/bin/jupyter lab'
```

### Option B: Use an Existing JupyterLab

If you already have JupyterLab installed (e.g. via Homebrew), register the environment as an additional kernel instead:

```bash
uv venv ~/.venvs/pampi
uv pip install --python ~/.venvs/pampi ipykernel matplotlib
~/.venvs/pampi/bin/python -m ipykernel install --user --name pampi --display-name "PAMPI (Python 3)"
```

In JupyterLab, open a notebook and choose **Kernel → Change Kernel… → PAMPI (Python 3)**.
The choice is stored in the notebook when you save it.

To remove the kernel again:

```bash
jupyter kernelspec uninstall pampi
```

### Verify the Setup

Open [`material/mpi-in-a-nutshell.ipynb`](material/mpi-in-a-nutshell.ipynb) and run the first cells.
The cell `%load_ext ice.magic` should print the temporary directory in use, and the following `%%mpi` cell should print one line per MPI process.
The ping-pong benchmark contains a plot cell, which needs `matplotlib`.

## Resetting a Notebook

Notebooks store the outputs of executed cells, and JupyterLab saves your edits automatically.
How to get back to a clean state depends on what you want to undo.

### Clear Outputs, Keep Your Changes

In JupyterLab, choose **Kernel → Restart Kernel and Clear Outputs of All Cells…**, then save the notebook (**File → Save Notebook**).
This also resets the kernel, so run the `%load_ext ice.magic` cell again before continuing.

From the terminal, the same works for one or all notebooks (close them in JupyterLab first):

```bash
~/.venvs/pampi/bin/jupyter nbconvert --clear-output --inplace material/mpi-in-a-nutshell.ipynb
~/.venvs/pampi/bin/jupyter nbconvert --clear-output --inplace material/*.ipynb
```

### Undo Your Changes

* **Since the last save:** **File → Revert Notebook to Checkpoint…** restores the state of the last manual save (<kbd>Ctrl</kbd>/<kbd>⌘</kbd>+<kbd>S</kbd>).
  JupyterLab keeps only this one checkpoint per notebook, in the hidden directory `.ipynb_checkpoints`.
* **Back to the original version:** delete the notebook (in the file browser or with `rm`) and run `make update`, which restores the original.
  (`git restore` only discards the changes since the last `make update`, because `make update` commits your changes before merging the new material.)

To keep the original intact in the first place, work on a copy: right-click the notebook in the file browser and choose **Duplicate**.
The copy must stay in the `material` directory, so that `%load_ext ice.magic` and the figures are found.

### Generated Files

Running the cells (re)creates the C sources in `material/code/` and stores the compiled program in `TMPDIR`.
The MPI I/O and the CFD notebooks write their files to `material/data/`, e.g. `io-fields.vtk`, which can be opened in ParaView.
Nothing needs to be cleaned up: every run overwrites these files, and `material/data/` can be deleted at any time.

## Repository Layout

| Path | Content |
|---|---|
| `overview.ipynb` | entry point with links to all notebooks |
| `material/*.ipynb` | the notebooks |
| `material/ice/magic.py` | IPython extension providing the `%%mpi` and `%%c` cell magics |
| `material/code/` | C sources of the examples, written when the notebook cells are run |
| `material/img/` | figures used in the notebooks |
| `material/data/` | files written by the MPI I/O and CFD notebooks (created when they run) |

## The `%%mpi` Cell Magic

Each notebook loads the extension with `%load_ext ice.magic`.
A `%%mpi` cell is treated as the body of `main()`, wrapped with `MPI_Init()`, `MPI_Finalize()` and the variables `rank` and `size`; `-r` takes the cell as a complete program instead.
The most important options are `-n N` (number of processes, default 4), `-d` (show generated code), `-v` (show commands), and `-T S` (timeout, default 60 s).
`-m` enables OpenMP for hybrid MPI+OpenMP code (and for serial `%%c` cells); set the number of threads per process with `-e OMP_NUM_THREADS=N`.
Run `%mpi -h` in a notebook for the full list.

The OpenMP compiler flags are detected automatically: `-fopenmp` works with GCC, LLVM clang and the Intel compilers.
Apple's clang, which `mpicc` uses on macOS by default, needs the OpenMP runtime from Homebrew (`brew install libomp`), which the extension finds by itself.
With Open MPI, `-m` also adds `--bind-to none` to the launcher flags, so that the threads of a process are not confined to a single core.

The launcher can be configured with environment variables, which must be set **before** starting JupyterLab:

| Variable | Default | Purpose |
|---|---|---|
| `ICE_MPI_LAUNCHER` | `mpirun` | program used to start MPI applications, e.g. `srun` |
| `ICE_MPI_LAUNCHER_FLAGS` | `--oversubscribe` for Open MPI, empty otherwise | default flags passed to the launcher |
| `ICE_OPENMP_FLAGS` | detected automatically | compiler flags enabling OpenMP for `-m`, e.g. `-qopenmp` |
| `TMPDIR` | `$HOME` if unset | where the generated source code and binary are stored |

For example, on a cluster node within a Slurm job:

```bash
ICE_MPI_LAUNCHER=srun ICE_MPI_LAUNCHER_FLAGS="" ~/.venvs/pampi/bin/jupyter lab
```

## Troubleshooting

* **`ModuleNotFoundError: No module named 'matplotlib'`** — the notebook runs on a kernel without matplotlib.
  Check the kernel name in the top right corner of the notebook and switch to the environment set up above.
* **`ModuleNotFoundError: No module named 'ice'`** — the kernel was not started in the `material` directory.
  Open the notebooks from the JupyterLab file browser, which starts each kernel in the directory of its notebook.
* **`[ICE] Compiler mpicc not found`** — MPI is not in the `PATH` of the JupyterLab process.
  Load the MPI module or fix the `PATH` in the shell, then restart JupyterLab from that shell.
* **`There are not enough slots available in the system`** — Open MPI refuses to start more processes than cores.
  The extension adds `--oversubscribe` automatically for Open MPI; if you set `ICE_MPI_LAUNCHER_FLAGS`, include it there.
* **`[ICE] No working OpenMP flags found`** — the compiler behind `mpicc` cannot compile OpenMP code.
  On macOS, run `brew install libomp`; otherwise set `ICE_OPENMP_FLAGS` to the flags of your compiler before starting JupyterLab.
* **A cell hangs** — the program probably deadlocked.
  It is killed after the timeout (`-T`, default 60 s); interrupting the kernel (■) also stops all MPI processes.
