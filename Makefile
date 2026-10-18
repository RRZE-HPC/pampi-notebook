# Makefile for the PAMPI notebooks
#
#   make init   create the virtual environment and install the packages
#   make run    start JupyterLab (default; runs init first if needed)
#   make clean  clear all notebook outputs, remove caches and checkpoints
#   make check  check that MPI, OpenMP and the environment work
#   make test   execute all notebooks headless and report failing cells
#   make update fetch new notebooks from the course repository, keep own changes
#
# The location of the environment can be changed, e.g.
#   make VENV=~/.venvs/pampi
# and the tests can be restricted to some notebooks, e.g.
#   make test NOTEBOOKS=material/collectives.ipynb

VENV    ?= .venv
PYTHON  ?= python3
JUPYTER := $(VENV)/bin/jupyter
STAMP   := $(VENV)/.installed
TESTDIR := .test

NOTEBOOKS := $(wildcard *.ipynb material/*.ipynb)

# use uv if available, plain venv + pip otherwise
UV := $(shell command -v uv 2>/dev/null)
ifdef UV
  MKVENV  = uv venv --python $(PYTHON) $(VENV)
  INSTALL = uv pip install --python $(VENV)/bin/python
else
  MKVENV  = $(PYTHON) -m venv $(VENV)
  INSTALL = $(VENV)/bin/pip install
endif

.DEFAULT_GOAL := run
.PHONY: init run clean check test update

init: $(STAMP)

$(STAMP): requirements.txt
	@test -x $(VENV)/bin/python || $(MKVENV)
	$(INSTALL) -r requirements.txt
	@touch $@

run: $(STAMP)
	$(JUPYTER) lab

clean: $(STAMP)
	@echo "Clearing outputs (close the notebooks in JupyterLab first, or it saves them again)"
	$(JUPYTER) nbconvert --log-level WARN --clear-output --inplace $(NOTEBOOKS)
	find . -path ./$(VENV) -prune -o \( -name .ipynb_checkpoints -o -name __pycache__ \) -type d -print -exec rm -rf {} +
	rm -rf material/data $(TESTDIR)

check:
	@sh tools/check.sh $(VENV)

test: $(STAMP)
	$(VENV)/bin/python tools/run_notebooks.py $(TESTDIR) $(NOTEBOOKS)

update: $(STAMP)
	@git rev-parse --is-inside-work-tree >/dev/null 2>&1 || \
	  { echo "This folder is not a git clone - get the material with git clone, see README.md"; exit 1; }
	$(VENV)/bin/gitpuller "$$(git remote get-url origin)" "$$(git rev-parse --abbrev-ref HEAD)" .

