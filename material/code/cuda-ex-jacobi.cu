#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <math.h>
#include <cuda-check.h>

// cell (i, j) of the grid with imax x jmax inner cells and one layer of boundary cells;
// i is the contiguous index, as in the Laplace solver of the previous notebooks
#define IDX(i, j) ((size_t)(j) * (imax + 2) + (i))

// one Jacobi sweep: thread (threadIdx.x, threadIdx.y) of block (blockIdx.x, blockIdx.y) updates one inner cell
__global__ void jacobi(double *unew, const double *u, int imax, int jmax) {
    // TODO: compute i and j (1 <= i <= imax, 1 <= j <= jmax) from the thread and block indices, then
    //       unew(i, j) = (u(i - 1, j) + u(i + 1, j) + u(i, j - 1) + u(i, j + 1)) / 4
}

int main(void) {
    const int imax = 4096, jmax = 4096;         // inner cells
    const size_t bytes = (size_t)(imax + 2) * (jmax + 2) * sizeof(double);
    const double h = 1.0 / (imax + 1);
    const int check = 10, sweeps = 1000;

    // initial values: u = x + y on the boundary, 0 inside
    double *u = (double *)malloc(bytes), *unew = (double *)malloc(bytes), *result = (double *)malloc(bytes);
    for (int j = 0; j <= jmax + 1; ++j)
        for (int i = 0; i <= imax + 1; ++i)
            u[IDX(i, j)] = unew[IDX(i, j)] = (i == 0 || j == 0 || i == imax + 1 || j == jmax + 1) ? (i + j) * h : 0.0;

    double *d_u, *d_unew;
    CUDA_CHECK(cudaMalloc(&d_u, bytes));
    CUDA_CHECK(cudaMalloc(&d_unew, bytes));
    CUDA_CHECK(cudaMemcpy(d_u, u, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_unew, u, bytes, cudaMemcpyHostToDevice));

    // reference: a few sweeps on the CPU
    for (int s = 0; s < check; ++s) {
        for (int j = 1; j <= jmax; ++j)
            for (int i = 1; i <= imax; ++i)
                unew[IDX(i, j)] = 0.25 * (u[IDX(i - 1, j)] + u[IDX(i + 1, j)] + u[IDX(i, j - 1)] + u[IDX(i, j + 1)]);
        double *tmp = u; u = unew; unew = tmp;
    }

    // the same sweeps on the GPU: 2D blocks of 32 x 8 threads, a 2D grid covering the inner cells
    const dim3 threads(32, 8);
    const dim3 blocks((imax + threads.x - 1) / threads.x, (jmax + threads.y - 1) / threads.y);
    for (int s = 0; s < check; ++s) {
        jacobi<<<blocks, threads>>>(d_unew, d_u, imax, jmax);
        double *tmp = d_u; d_u = d_unew; d_unew = tmp;
    }
    CUDA_CHECK_LAUNCH();
    CUDA_CHECK(cudaMemcpy(result, d_u, bytes, cudaMemcpyDeviceToHost));
    double diff = 0.0;
    for (size_t k = 0; k < bytes / sizeof(double); ++k)
        diff = fmax(diff, fabs(result[k] - u[k]));
    printf("%d sweeps: max. difference to the CPU: %.2e\n", check, diff);

    // performance
    GpuTimer timer;
    timer.start();
    for (int s = 0; s < sweeps; ++s) {
        jacobi<<<blocks, threads>>>(d_unew, d_u, imax, jmax);
        double *tmp = d_u; d_u = d_unew; d_unew = tmp;
    }
    float ms = timer.stop();
    CUDA_CHECK_LAUNCH();
    printf("%d sweeps: %.1f ms, %.0f cell updates per ns, %.0f GB/s (16 bytes per update)\n",
           sweeps, ms, (double)imax * jmax * sweeps / (ms * 1e6), 16.0 * imax * jmax * sweeps / (ms * 1e6));

    CUDA_CHECK(cudaFree(d_u));
    CUDA_CHECK(cudaFree(d_unew));
    free(u);
    free(unew);
    free(result);
    return 0;
}
