#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <cuda-check.h>

// the number of the SM this thread runs on: a special register, read with inline PTX assembly
__device__ unsigned sm_id(void) {
    unsigned id;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(id));
    return id;
}

// a kernel: executed by every thread of the grid
__global__ void hello(void) {
    unsigned warp = threadIdx.x / warpSize, lane = threadIdx.x % warpSize;
    if (lane == 0)                              // one line per warp
        printf("block %u, warp %u (threads %3u-%3u) runs on SM %u\n",
               blockIdx.x, warp, threadIdx.x, threadIdx.x + warpSize - 1, sm_id());
}

int main(void) {
    hello<<<4, 96>>>();                         // a grid of 4 blocks of 96 threads (3 warps) each
    CUDA_CHECK_LAUNCH();
    printf("host: kernel launched\n");
    CUDA_CHECK(cudaDeviceSynchronize());        // wait for the kernel - this also flushes its output
    printf("host: kernel finished\n");
    return 0;
}
