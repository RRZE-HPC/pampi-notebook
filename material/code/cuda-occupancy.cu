#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <cuda-check.h>

__global__ void triad(double *a, const double *b, const double *c, double s, int64_t n) {
    int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (i < n)
        a[i] = b[i] + s * c[i];
}

int main(void) {
    int device, threadsSM;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaDeviceGetAttribute(&threadsSM, cudaDevAttrMaxThreadsPerMultiProcessor, device));
    const int max_warps = threadsSM / 32;

    // the resources of the compiled kernel
    cudaFuncAttributes attr;
    CUDA_CHECK(cudaFuncGetAttributes(&attr, triad));
    printf("triad: %d registers per thread, %zu bytes of shared memory per block\n\n", attr.numRegs, attr.sharedSizeBytes);

    // how many blocks fit on one SM, depending on the block size and the shared memory per block?
    printf("threads/block  shared memory/block  blocks/SM  warps/SM  occupancy\n");
    const size_t smem[] = { 0, 16 * 1024, 48 * 1024 };
    for (int s = 0; s < 3; ++s)
        for (int threads = 32; threads <= 1024; threads *= 4) {
            int blocks;
            CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, triad, threads, smem[s]));
            const int warps = blocks * threads / 32;
            printf("%13d  %15zu KiB  %9d  %8d  %8.0f %%\n", threads, smem[s] / 1024, blocks, warps, 100.0 * warps / max_warps);
        }
    return 0;
}
