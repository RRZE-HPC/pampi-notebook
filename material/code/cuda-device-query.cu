#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <cuda-check.h>

// one property of the GPU, queried by its attribute
static int attribute(cudaDeviceAttr attr, int device) {
    int value;
    CUDA_CHECK(cudaDeviceGetAttribute(&value, attr, device));
    return value;
}

int main(void) {
    int count, dev = 0;
    CUDA_CHECK(cudaGetDeviceCount(&count));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
    printf("%d GPU(s), device %d: %s, compute capability %d.%d\n\n", count, dev, prop.name,
           attribute(cudaDevAttrComputeCapabilityMajor, dev), attribute(cudaDevAttrComputeCapabilityMinor, dev));

    const int sms       = attribute(cudaDevAttrMultiProcessorCount, dev);
    const int warp      = attribute(cudaDevAttrWarpSize, dev);
    const int threadsSM = attribute(cudaDevAttrMaxThreadsPerMultiProcessor, dev);
    const int blocksSM  = attribute(cudaDevAttrMaxBlocksPerMultiprocessor, dev);
    const int regsSM    = attribute(cudaDevAttrMaxRegistersPerMultiprocessor, dev);
    const int smemSM    = attribute(cudaDevAttrMaxSharedMemoryPerMultiprocessor, dev);
    const int l2        = attribute(cudaDevAttrL2CacheSize, dev);
    const int clock     = attribute(cudaDevAttrClockRate, dev);              // kHz
    const int memClock  = attribute(cudaDevAttrMemoryClockRate, dev);        // kHz
    const int busWidth  = attribute(cudaDevAttrGlobalMemoryBusWidth, dev);   // bits

    printf("%-34s %s\n", "GPU term", "value  (CPU counterpart)");
    printf("%-34s %d  (cores)\n", "streaming multiprocessors (SMs)", sms);
    printf("%-34s %d threads  (SIMD width)\n", "warp size", warp);
    printf("%-34s %d threads = %d warps  (hardware threads per core: 1-2)\n", "resident threads per SM", threadsSM, threadsSM / warp);
    printf("%-34s %d\n", "resident blocks per SM", blocksSM);
    printf("%-34s %d\n", "threads per block (max.)", attribute(cudaDevAttrMaxThreadsPerBlock, dev));
    printf("%-34s %d x 32 bit = %d KiB  (a few KiB per core)\n", "registers per SM", regsSM, regsSM * 4 / 1024);
    printf("%-34s %d KiB  (no counterpart; L1 cache: 32-64 KiB)\n", "shared memory per SM (max.)", smemSM / 1024);
    printf("%-34s %.1f MiB  (L2 + L3 cache)\n", "L2 cache", l2 / 1048576.0);
    printf("%-34s %.1f GiB  (main memory)\n", "global memory", prop.totalGlobalMem / 1073741824.0);
    printf("%-34s %.0f GB/s  (bus %d bit, %d MHz, double data rate)\n", "memory bandwidth (peak)",
           2.0 * memClock * 1e3 * (busWidth / 8) / 1e9, busWidth, memClock / 1000);
    printf("%-34s %d MHz\n", "clock (max.)", clock / 1000);
    printf("%-34s %d  (hardware threads of the CPU)\n", "threads in flight (max.)", sms * threadsSM);

    // the same numbers in the form used by the Python cells of this notebook
    printf("\nfor the Little's-law cell (replace the latency by your measurement):\n");
    printf("    'my GPU': dict(bandwidth=%.0fe9, latency=500e-9, units=%d, threads=%d),\n",
           2.0 * memClock * 1e3 * (busWidth / 8) / 1e9, sms, sms * threadsSM);
    printf("for the occupancy calculator:\n");
    printf("    limits = dict(max_warps=%d, max_blocks=%d, regs_per_sm=%d, smem_per_sm=%d)\n",
           threadsSM / warp, blocksSM, regsSM, smemSM);
    return 0;
}
