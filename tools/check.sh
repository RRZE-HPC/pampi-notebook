#!/bin/sh
# Checks that MPI, OpenMP and the Python environment work for the notebooks.
#
# usage: check.sh [VENV]

venv=${1:-.venv}
launcher=${ICE_MPI_LAUNCHER:-mpirun}
failed=0

ok()   { printf '  ok    %s\n' "$*"; }
warn() { printf '  warn  %s\n' "$*"; }
fail() { printf '  FAIL  %s\n' "$*"; failed=1; }

tmp=$(mktemp -d) || exit 1
trap 'rm -rf "$tmp"' EXIT

echo "MPI"
if command -v mpicc >/dev/null 2>&1; then
    ok "mpicc: $(command -v mpicc) ($(mpicc --version 2>&1 | head -n 1))"
else
    fail "mpicc not found - install MPI or load the MPI module"
fi
if command -v "$launcher" >/dev/null 2>&1; then
    ok "$launcher: $(command -v "$launcher")"
else
    fail "$launcher not found - install MPI or load the MPI module"
fi

if [ $failed -eq 0 ]; then
    # the same default launcher flags as the ICE magic
    if [ -n "${ICE_MPI_LAUNCHER_FLAGS+set}" ]; then
        flags=$ICE_MPI_LAUNCHER_FLAGS
    elif "$launcher" --version 2>&1 | grep -q 'Open MPI'; then
        flags=--oversubscribe
    else
        flags=
    fi
    cat > "$tmp/hello.c" <<'EOF'
#include <stdio.h>
#include <mpi.h>
int main(int argc, char **argv) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    printf("rank %d of %d\n", rank, size);
    MPI_Finalize();
    return 0;
}
EOF
    if ! mpicc -o "$tmp/hello" "$tmp/hello.c" > "$tmp/log" 2>&1; then
        fail "mpicc cannot compile an MPI program:"
        sed 's/^/        /' "$tmp/log"
    # shellcheck disable=SC2086
    elif ! "$launcher" $flags -n 2 "$tmp/hello" > "$tmp/log" 2>&1; then
        fail "$launcher $flags -n 2 failed:"
        sed 's/^/        /' "$tmp/log"
    elif [ "$(grep -c 'of 2$' "$tmp/log")" -ne 2 ]; then
        fail "$launcher $flags -n 2 did not start 2 processes:"
        sed 's/^/        /' "$tmp/log"
    else
        ok "compiled and ran an MPI program with 2 processes"
    fi

    echo "OpenMP (only needed for the hybrid MPI+OpenMP notebook)"
    printf '#include <omp.h>\nint main(void) { int n = 0;\n#pragma omp parallel reduction(+:n)\nn += 1;\nreturn n < 1; }\n' > "$tmp/omp.c"
    try_omp() {
        mpicc "$@" -o "$tmp/omp" "$tmp/omp.c" $libs > /dev/null 2>&1 && "$tmp/omp"
    }
    if [ -n "$ICE_OPENMP_FLAGS" ]; then
        # shellcheck disable=SC2086
        libs= ; if try_omp $ICE_OPENMP_FLAGS; then ok "ICE_OPENMP_FLAGS=$ICE_OPENMP_FLAGS"
        else fail "ICE_OPENMP_FLAGS=$ICE_OPENMP_FLAGS does not work with mpicc"; fi
    else
        libs= ; found=
        if try_omp -fopenmp; then found=-fopenmp; fi
        if [ -z "$found" ]; then
            for p in $(brew --prefix libomp 2>/dev/null) /opt/homebrew/opt/libomp /usr/local/opt/libomp; do
                [ -d "$p/include" ] || continue
                libs=-lomp
                if try_omp -Xpreprocessor -fopenmp "-I$p/include" "-L$p/lib"; then found="libomp from $p"; break; fi
            done
        fi
        if [ -n "$found" ]; then ok "OpenMP works ($found)"
        else warn "no working OpenMP flags - on macOS: brew install libomp, otherwise set ICE_OPENMP_FLAGS"; fi
    fi
fi

echo "Python environment"
if [ -x "$venv/bin/python" ]; then
    if out=$("$venv/bin/python" -c 'import jupyterlab, matplotlib, numpy, sys; print(sys.version.split()[0])' 2>&1); then
        ok "$venv: Python $out with jupyterlab, matplotlib, numpy"
    else
        fail "$venv is missing packages - run make init"
        printf '%s\n' "$out" | tail -n 1 | sed 's/^/        /'
    fi
else
    fail "no environment in $venv - run make init"
fi

echo
if [ $failed -eq 0 ]; then echo "All checks passed."; else echo "Some checks failed."; fi
exit $failed
