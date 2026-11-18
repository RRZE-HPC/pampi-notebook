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

echo "CUDA (only needed for the CUDA notebook, requires an NVIDIA GPU)"
if command -v nvcc >/dev/null 2>&1; then
    cat > "$tmp/cuda.cu" <<'EOF'
#include <stdio.h>
__global__ void add(int *x) { atomicAdd(x, 1); }
int main(void) {
    int *x, n = 0;
    if (cudaMalloc(&x, sizeof(int)) != cudaSuccess) return 1;
    cudaMemcpy(x, &n, sizeof(int), cudaMemcpyHostToDevice);
    add<<<4, 64>>>(x);
    cudaMemcpy(&n, x, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(x);
    return n != 256;
}
EOF
    if ! nvcc -arch=native -o "$tmp/cuda" "$tmp/cuda.cu" > "$tmp/log" 2>&1; then
        warn "nvcc cannot compile a CUDA program (is there a GPU on this node?):"
        sed 's/^/        /' "$tmp/log"
    elif ! "$tmp/cuda"; then
        warn "a CUDA program compiled, but did not run correctly - is there a GPU on this node?"
    else
        ok "compiled and ran a CUDA kernel ($(nvcc --version | tail -n 1))"
    fi
else
    warn "nvcc not found - the CUDA cells will be skipped (on a cluster: load the CUDA module on a GPU node)"
fi

echo "MPI+CUDA (only needed for the MPI+CUDA notebook, requires an NVIDIA GPU)"
cxx=${ICE_MPI_CXX:-mpicxx}
if ! command -v nvcc >/dev/null 2>&1; then
    warn "nvcc not found - the MPI+CUDA cells will be skipped"
elif ! command -v "$cxx" >/dev/null 2>&1; then
    warn "$cxx not found - set ICE_MPI_CXX to the MPI C++ compiler wrapper, e.g. mpiicpx"
elif [ $failed -ne 0 ]; then
    warn "skipped, MPI does not work"
else
    cat > "$tmp/mpicuda.cu" <<'EOF'
#include <stdio.h>
#include <mpi.h>
#if defined(OPEN_MPI) && OPEN_MPI
#include <mpi-ext.h>
#endif
int main(int argc, char **argv) {
    int n = 0;
    MPI_Init(&argc, &argv);
    if (cudaGetDeviceCount(&n) != cudaSuccess || n < 1) {
        MPI_Finalize();
        return 1;
    }
#if defined(MPIX_CUDA_AWARE_SUPPORT)
    printf("%d GPU(s), MPI library %s\n", n, MPIX_Query_cuda_support() ? "is CUDA-aware" : "is not CUDA-aware");
#else
    printf("%d GPU(s), CUDA-awareness of the MPI library unknown\n", n);
#endif
    MPI_Finalize();
    return 0;
}
EOF
    # shellcheck disable=SC2086
    if ! nvcc -ccbin "$cxx" -arch=native -o "$tmp/mpicuda" "$tmp/mpicuda.cu" > "$tmp/log" 2>&1; then
        warn "nvcc -ccbin $cxx cannot compile an MPI+CUDA program:"
        sed 's/^/        /' "$tmp/log"
    elif ! "$launcher" $flags -n 1 "$tmp/mpicuda" > "$tmp/log" 2>&1; then
        warn "an MPI+CUDA program compiled, but did not run - is there a GPU on this node?"
        sed 's/^/        /' "$tmp/log"
    else
        ok "compiled and ran an MPI+CUDA program: $(tail -n 1 "$tmp/log")"
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
