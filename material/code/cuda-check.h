// Helpers for the CUDA examples of the PAMPI notebooks.
#pragma once
#include <stdio.h>
#include <stdlib.h>

// Every CUDA runtime call returns an error code - check it. Kernel launches return nothing,
// their errors (e.g. too many threads per block) are fetched with cudaGetLastError().
#define CUDA_CHECK(call)                                                            \
    do {                                                                            \
        cudaError_t err_ = (call);                                                  \
        if (err_ != cudaSuccess) {                                                  \
            fprintf(stderr, "%s:%d: %s failed: %s\n", __FILE__, __LINE__, #call,    \
                    cudaGetErrorString(err_));                                      \
            exit(1);                                                                \
        }                                                                           \
    } while (0)
#define CUDA_CHECK_LAUNCH() CUDA_CHECK(cudaGetLastError())

// Times GPU work with two events, which the GPU timestamps when it reaches them in the stream:
//     GpuTimer timer;
//     timer.start();
//     kernel<<<blocks, threads>>>(...);
//     float ms = timer.stop();     // waits until the kernel has finished
struct GpuTimer {
    cudaEvent_t begin, end;
    GpuTimer()  { CUDA_CHECK(cudaEventCreate(&begin)); CUDA_CHECK(cudaEventCreate(&end)); }
    ~GpuTimer() { cudaEventDestroy(begin); cudaEventDestroy(end); }
    void start(cudaStream_t stream = 0) { CUDA_CHECK(cudaEventRecord(begin, stream)); }
    float stop(cudaStream_t stream = 0) {
        CUDA_CHECK(cudaEventRecord(end, stream));
        CUDA_CHECK(cudaEventSynchronize(end));
        float ms;
        CUDA_CHECK(cudaEventElapsedTime(&ms, begin, end));
        return ms;
    }
};

// the number of SMs of the current GPU
static inline int sm_count(void) {
    int device, count;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaDeviceGetAttribute(&count, cudaDevAttrMultiProcessorCount, device));
    return count;
}
