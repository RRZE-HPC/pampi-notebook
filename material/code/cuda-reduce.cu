#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <cuda-check.h>

// the sum over the 32 threads of a warp: a binary tree in registers, without shared memory;
// __shfl_down_sync() reads the variable v of the thread offset lanes further up.
// The mask 0xffffffff says that all 32 lanes take part: the block size must be a multiple of 32.
__device__ double warp_sum(double v) {
    for (int offset = 16; offset > 0; offset /= 2)
        v += __shfl_down_sync(0xffffffff, v, offset);
    return v;                                   // the complete sum is in lane 0
}

__global__ void sum(const double *x, int64_t n, double *result) {
    __shared__ double partial[32];              // one value per warp of the block (at most 32 warps)
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;

    // 1. every thread: a private partial sum, with a grid-stride loop
    double v = 0.0;
    for (int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; i < n; i += (int64_t)gridDim.x * blockDim.x)
        v += x[i];

    // 2. every warp: tree in registers, lane 0 stores the result of its warp in shared memory
    v = warp_sum(v);
    if (lane == 0)
        partial[warp] = v;
    __syncthreads();                            // barrier of the block: all partial sums are stored

    // 3. the first warp: sum of the partial sums of all warps of the block
    if (warp == 0) {
        v = lane < (int)(blockDim.x / 32) ? partial[lane] : 0.0;
        v = warp_sum(v);
        if (lane == 0)
            atomicAdd(result, v);               // 4. one atomic update per block
    }
}

int main(void) {
    const int64_t n = 1 << 25;
    double *x = (double *)malloc(n * sizeof(double)), *d_x, *d_result;
    for (int64_t i = 0; i < n; ++i)
        x[i] = 1.0 / (i + 1);
    CUDA_CHECK(cudaMalloc(&d_x, n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_result, sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_x, x, n * sizeof(double), cudaMemcpyHostToDevice));

    double reference = 0.0;                     // the CPU result, summed in order
    for (int64_t i = 0; i < n; ++i)
        reference += x[i];

    const int threads = 256, blocks = 4 * sm_count();
    GpuTimer timer;
    for (int run = 0; run < 3; ++run) {
        double result;
        CUDA_CHECK(cudaMemset(d_result, 0, sizeof(double)));
        timer.start();
        sum<<<blocks, threads>>>(d_x, n, d_result);
        float ms = timer.stop();
        CUDA_CHECK_LAUNCH();
        CUDA_CHECK(cudaMemcpy(&result, d_result, sizeof(double), cudaMemcpyDeviceToHost));
        printf("GPU: %.17f  (CPU - GPU = %9.2e)  %.3f ms, %.0f GB/s\n",
               result, reference - result, ms, n * sizeof(double) / ms / 1e6);
    }
    printf("CPU: %.17f\n", reference);

    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_result));
    free(x);
    return 0;
}
