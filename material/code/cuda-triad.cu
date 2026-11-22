#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <cuda-check.h>

// one thread per element: the loop body of a[i] = b[i] + s * c[i]
__global__ void triad(double *a, const double *b, const double *c, double s, int64_t n) {
    int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;     // global thread index
    if (i < n)                                  // the last block may stick out of the array
        a[i] = b[i] + s * c[i];
}

int main(void) {
    const int64_t n = 1 << 25;                  // 32 Mi elements: 256 MiB per array
    const size_t bytes = n * sizeof(double);

    // host memory: ordinary malloc
    double *a = (double *)malloc(bytes), *b = (double *)malloc(bytes), *c = (double *)malloc(bytes);
    for (int64_t i = 0; i < n; ++i) {
        a[i] = 0.0;                             // touch a, too: the copy back should not measure page faults
        b[i] = 1.0;
        c[i] = 2.0;
    }

    // device memory: a separate address space - d_a cannot be dereferenced on the host
    double *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_c, bytes));

    GpuTimer timer;
    timer.start();
    CUDA_CHECK(cudaMemcpy(d_b, b, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_c, c, bytes, cudaMemcpyHostToDevice));
    float ms_in = timer.stop();

    const int threads = 256;                    // threads per block
    const int64_t blocks = (n + threads - 1) / threads;
    triad<<<blocks, threads>>>(d_a, d_b, d_c, 3.0, n);      // warm-up: the first launch has extra costs
    CUDA_CHECK_LAUNCH();

    const int reps = 10;
    timer.start();
    for (int r = 0; r < reps; ++r)
        triad<<<blocks, threads>>>(d_a, d_b, d_c, 3.0, n);
    float ms_kernel = timer.stop() / reps;

    timer.start();
    CUDA_CHECK(cudaMemcpy(a, d_a, bytes, cudaMemcpyDeviceToHost));
    float ms_out = timer.stop();

    int64_t wrong = 0;
    for (int64_t i = 0; i < n; ++i)
        wrong += a[i] != 7.0;
    printf("%lld blocks of %d threads, %lld wrong results\n", (long long)blocks, threads, (long long)wrong);
    printf("copy b, c to the GPU: %7.2f ms  %7.1f GB/s\n", ms_in, 2 * bytes / ms_in / 1e6);
    printf("triad kernel        : %7.2f ms  %7.1f GB/s\n", ms_kernel, 3 * bytes / ms_kernel / 1e6);
    printf("copy a to the host  : %7.2f ms  %7.1f GB/s\n", ms_out, bytes / ms_out / 1e6);

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
    free(a);
    free(b);
    free(c);
    return 0;
}
