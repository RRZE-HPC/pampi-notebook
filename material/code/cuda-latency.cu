#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <cuda-check.h>

// a random-looking cycle through all n elements: next[i] = (a * i + c) mod n is a single cycle
// for n = 2^k, odd c and a = 1 mod 4 (a full-period linear congruential generator)
__global__ void init(uint32_t *next, uint32_t n) {
    for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
        next[i] = (1664525u * i + 1013904223u) & (n - 1);
}

// a hash: random starting points for the threads
__device__ uint32_t scramble(uint32_t x) {
    x ^= x >> 16;
    x *= 0x45d9f3bu;
    x ^= x >> 16;
    return x;
}

// every thread follows its own chain of dependent loads: the address of a load is the result of the previous one
__global__ void chase(const uint32_t *next, uint32_t n, int steps, uint32_t *out) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t p = scramble(tid) & (n - 1);
    for (int s = 0; s < steps; ++s)
        p = next[p];
    out[tid] = p;                               // keeps the compiler from removing the loop
}

int main(void) {
    int device, threadsSM, l2;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaDeviceGetAttribute(&threadsSM, cudaDevAttrMaxThreadsPerMultiProcessor, device));
    CUDA_CHECK(cudaDeviceGetAttribute(&l2, cudaDevAttrL2CacheSize, device));
    const int sms = sm_count(), max_warps = threadsSM / 32;

    // the table must be much larger than the L2 cache, so that almost every load goes to memory:
    // at least 256 MiB, doubled up to 16x the L2 cache (but at most a quarter of the free memory)
    size_t free_mem, total_mem;
    CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));
    uint32_t n = 1u << 26;                      // elements of 4 bytes, a power of two
    while ((size_t)n * 4 < 16 * (size_t)l2 && (size_t)n * 8 <= free_mem / 4 && n < (1u << 30))
        n *= 2;
    printf("table: %.0f MiB, L2 cache: %.0f MiB\n", n * 4.0 / 1048576, l2 / 1048576.0);

    uint32_t *next, *out;
    CUDA_CHECK(cudaMalloc(&next, n * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&out, (size_t)sms * threadsSM * sizeof(uint32_t)));
    init<<<1024, 256>>>(next, n);
    CUDA_CHECK_LAUNCH();

    // 1. a single thread: every load waits for the previous one - pure latency
    GpuTimer timer;
    const int steps1 = 100000;
    chase<<<1, 1>>>(next, n, 1000, out);        // warm-up
    CUDA_CHECK_LAUNCH();
    timer.start();
    chase<<<1, 1>>>(next, n, steps1, out);
    const double latency = timer.stop() * 1e6 / steps1;     // ns per load
    printf("1 thread: %.0f ns per dependent load\n\n", latency);

    // 2. w warps per SM, every thread with its own chain: the SM switches to another warp
    //    whenever one waits for its load. Every thread has one load in flight at any time, and
    //    Little's law gives the latency under load: loads in flight / throughput
    const int steps = 2000;
    printf("warps/SM  loads in flight (= threads)  loads/ns  latency [ns]\n");
    for (int w = 1; w <= max_warps; w *= 2) {
        const int threads = w == 1 ? 32 : 64;   // threads per block
        const int blocks = sms * w * 32 / threads;
        timer.start();
        chase<<<blocks, threads>>>(next, n, steps, out);
        const float ms = timer.stop();
        const double rate = (double)blocks * threads * steps / (ms * 1e6);
        printf("%8d  %28d  %8.1f  %12.0f\n", w, blocks * threads, rate, blocks * threads / rate);
    }

    CUDA_CHECK(cudaFree(next));
    CUDA_CHECK(cudaFree(out));
    return 0;
}
