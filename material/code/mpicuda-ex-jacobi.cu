#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <mpi.h>
#include <math.h>
#include <mpicuda.h>

// cell (i, j) of the local grid with imax x jmaxLocal inner cells and one layer of halo or
// boundary cells; i is the contiguous index, the grid is distributed in j: the halos are rows
#define IDX(i, j) ((size_t)(j) * (imax + 2) + (i))

// one Jacobi sweep, as in the CUDA primer
__global__ void jacobi(double *unew, const double *u, int imax, int jmax) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    const int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    if (i <= imax && j <= jmax)
        unew[IDX(i, j)] = 0.25 * (u[IDX(i - 1, j)] + u[IDX(i + 1, j)] + u[IDX(i, j - 1)] + u[IDX(i, j + 1)]);
}

// exchanges the halo rows of d_u with the neighbors below and above (MPI_PROC_NULL at the boundary)
static void halo_exchange(double *d_u, double *h_send, double *h_recv, int imax, int jmaxLocal,
                          int below, int above, bool aware) {
    // TODO: send the top inner row (j = jmaxLocal) up and receive the bottom halo (j = 0) from below,
    //       then send the bottom inner row (j = 1) down and receive the top halo (j = jmaxLocal + 1)
    //       from above, each with one MPI_Sendrecv() of the imax values i = 1 ... imax:
    //       aware: directly from and to the rows in GPU memory
    //       else:  staged through the pinned host buffers h_send and h_recv (imax values each)
    //       Make sure that the GPU has finished the previous sweep before MPI reads d_u!
}

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    select_gpu();
    const char *how;
    const bool aware = mpi_cuda_aware(&how) == 1;

    const int imax = 2048, jmax = 2048;             // global inner cells
    const int sweeps = 2000;                        // a fixed number, so that runtimes are comparable
    const double h = 1.0 / (imax + 1);

    // the rows of this process, and its neighbors
    const int jmaxLocal = jmax / size + (rank < jmax % size ? 1 : 0);
    const int joff = rank * (jmax / size) + (rank < jmax % size ? rank : jmax % size);
    const int below = rank > 0 ? rank - 1 : MPI_PROC_NULL, above = rank < size - 1 ? rank + 1 : MPI_PROC_NULL;
    const size_t bytes = (size_t)(imax + 2) * (jmaxLocal + 2) * sizeof(double);

    // initial values: u = x + y on the global boundary, 0 inside; the exact solution is x + y
    double *u = (double *)malloc(bytes);
    for (int j = 0; j <= jmaxLocal + 1; ++j)
        for (int i = 0; i <= imax + 1; ++i) {
            const int jg = joff + j;                // the global row
            u[IDX(i, j)] = (i == 0 || i == imax + 1 || jg == 0 || jg == jmax + 1) ? (i + jg) * h : 0.0;
        }

    double *d_u, *d_unew, *h_send, *h_recv;
    CUDA_CHECK(cudaMalloc(&d_u, bytes));
    CUDA_CHECK(cudaMalloc(&d_unew, bytes));
    CUDA_CHECK(cudaMemcpy(d_u, u, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_unew, u, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMallocHost(&h_send, imax * sizeof(double)));
    CUDA_CHECK(cudaMallocHost(&h_recv, imax * sizeof(double)));

    const dim3 threads(32, 8);
    const dim3 blocks((imax + threads.x - 1) / threads.x, (jmaxLocal + threads.y - 1) / threads.y);

    // the first messages set up the paths of the library (e.g. CUDA IPC), which can take several
    // 100 ms: exchange once before the time is taken - this also fills the halos for the first sweep
    halo_exchange(d_u, h_send, h_recv, imax, jmaxLocal, below, above, aware);
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    for (int s = 0; s < sweeps; ++s) {
        halo_exchange(d_u, h_send, h_recv, imax, jmaxLocal, below, above, aware);
        jacobi<<<blocks, threads>>>(d_unew, d_u, imax, jmaxLocal);
        CUDA_CHECK_LAUNCH();
        double *tmp = d_u; d_u = d_unew; d_unew = tmp;
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    const double t = MPI_Wtime() - start;

    // the mean distance to the exact solution: the same for every number of processes
    CUDA_CHECK(cudaMemcpy(u, d_u, bytes, cudaMemcpyDeviceToHost));
    double local_err = 0.0, err;
    for (int j = 1; j <= jmaxLocal; ++j)
        for (int i = 1; i <= imax; ++i)
            local_err += fabs(u[IDX(i, j)] - (i + joff + j) * h);
    MPI_Reduce(&local_err, &err, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    if (rank == 0)
        printf("%d processes, %s: %d sweeps in %.1f ms, %.1f cell updates per ns, mean error %.10e\n", size,
               aware ? "CUDA-aware" : "staged", sweeps, 1e3 * t, (double)imax * jmax * sweeps / (t * 1e9), err / ((double)imax * jmax));

    CUDA_CHECK(cudaFree(d_u));
    CUDA_CHECK(cudaFree(d_unew));
    CUDA_CHECK(cudaFreeHost(h_send));
    CUDA_CHECK(cudaFreeHost(h_recv));
    free(u);
    MPI_Finalize();
    return 0;
}
