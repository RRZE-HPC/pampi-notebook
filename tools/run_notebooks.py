"""Executes notebooks headless and reports cells that failed.

The ICE magic does not raise exceptions, it prints messages starting with [ICE] instead;
these are reported as failures, as are Python exceptions. Cells tagged `expect-ice-error`
are expected to produce such a message (e.g. a deliberate deadlock) and fail if they do not.

The executed notebooks are written to OUTDIR, the originals stay unchanged.

usage: python run_notebooks.py OUTDIR NOTEBOOK...
"""

import os
import pathlib
import sys
import tempfile
import time

import nbformat
from nbclient import NotebookClient

# messages of the ICE magic indicating that something went wrong
ICE_ERRORS = (
    '[ICE] Compiler ',
    '[ICE] Compilation failed',
    '[ICE] No working OpenMP flags',
    '[ICE] Execution killed',
    '[ICE] Execution finished with exit code',
)
EXPECT_TAG = 'expect-ice-error'
CELL_TIMEOUT = 900


def output_text(cell):
    text = []
    for out in cell.get('outputs', []):
        if out.output_type == 'stream':
            text.append(out.text)
        elif out.output_type == 'error':
            text.append(f'{out.ename}: {out.evalue}')
        elif 'text/plain' in out.get('data', {}):
            text.append(out.data['text/plain'])
    return ''.join(text)


def check(nb):
    """returns a list of (cell number, message) for all failed cells"""
    failures = []
    for i, cell in enumerate(nb.cells):
        if cell.cell_type != 'code':
            continue
        text = output_text(cell)
        errors = [l.strip() for l in text.splitlines() if l.strip().startswith(ICE_ERRORS)]
        errors += [f'{o.ename}: {o.evalue}' for o in cell.get('outputs', []) if o.output_type == 'error']
        expected = EXPECT_TAG in cell.metadata.get('tags', [])
        if expected and not errors:
            failures.append((i, f'expected an [ICE] error (tag {EXPECT_TAG}), but the cell succeeded'))
        elif not expected:
            failures += [(i, e) for e in errors]
    return failures


def run(path, outdir):
    nb = nbformat.read(path, as_version=4)
    client = NotebookClient(nb, timeout=CELL_TIMEOUT, allow_errors=True,
                            resources={'metadata': {'path': str(path.parent)}})
    start = time.monotonic()
    # the ICE magic always compiles to TMPDIR/code.c and TMPDIR/app; a private TMPDIR keeps
    # the test from clashing with notebooks running in JupyterLab at the same time
    tmpdir = tempfile.TemporaryDirectory(prefix='pampi-test-')
    env = dict(os.environ, TMPDIR=tmpdir.name)
    try:
        # the kernel's own stderr (startup warnings etc.) would clutter the report
        with open(outdir / 'kernel.log', 'a') as log:
            client.execute(env=env, stderr=log)
    except Exception as e:  # with allow_errors only raised for cell timeouts and dead kernels
        return [(-1, f'{type(e).__name__}: {str(e).strip().splitlines()[0] if str(e).strip() else ""}')], time.monotonic() - start
    finally:
        tmpdir.cleanup()
        out = outdir / path
        out.parent.mkdir(parents=True, exist_ok=True)
        nbformat.write(nb, out)
    return check(nb), time.monotonic() - start


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__.strip().splitlines()[-1])
    outdir = pathlib.Path(sys.argv[1])
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / 'kernel.log').unlink(missing_ok=True)
    failed = []
    for name in sys.argv[2:]:
        path = pathlib.Path(name)
        print(f'{name} ...', end=' ', flush=True)
        failures, seconds = run(path, outdir)
        print(f'{"FAILED" if failures else "ok"} ({seconds:.0f} s)')
        for cell, message in failures:
            print(f'    cell {cell}: {message}')
        if failures:
            failed.append(name)
    print(f'\n{len(sys.argv) - 2 - len(failed)} passed, {len(failed)} failed; executed notebooks in {outdir}/')
    sys.exit(1 if failed else 0)


if __name__ == '__main__':
    main()
