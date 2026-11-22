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
    const int64_t n = 1 << 24;                  // 128 MiB per array
    const size_t bytes = n * sizeof(double);
    const int nchunks = 8, nstreams = 4;
    const int64_t chunk = n / nchunks;

    // pinned (page-locked) host memory: the copy engines of the GPU can access it directly
    double *a, *b, *c, *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMallocHost(&a, bytes));
    CUDA_CHECK(cudaMallocHost(&b, bytes));
    CUDA_CHECK(cudaMallocHost(&c, bytes));
    for (int64_t i = 0; i < n; ++i) {
        b[i] = 1.0;
        c[i] = 2.0;
    }
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_c, bytes));
    GpuTimer timer;

    // for comparison: a copy from ordinary (pageable) memory goes through a pinned staging buffer
    double *pageable = (double *)malloc(bytes);
    for (int64_t i = 0; i < n; ++i)
        pageable[i] = 1.0;
    timer.start();
    CUDA_CHECK(cudaMemcpy(d_b, pageable, bytes, cudaMemcpyHostToDevice));
    float ms = timer.stop();
    printf("copy to GPU from pageable memory: %5.1f GB/s\n", bytes / ms / 1e6);
    timer.start();
    CUDA_CHECK(cudaMemcpy(d_b, b, bytes, cudaMemcpyHostToDevice));
    ms = timer.stop();
    printf("copy to GPU from pinned memory  : %5.1f GB/s\n\n", bytes / ms / 1e6);
    free(pageable);

    triad<<<(n + 255) / 256, 256>>>(d_a, d_b, d_c, 3.0, n);    // warm-up: the first launch has extra costs
    CUDA_CHECK_LAUNCH();

    // 1. everything in one stream: copy in, compute, copy out - one after the other
    timer.start();
    CUDA_CHECK(cudaMemcpyAsync(d_b, b, bytes, cudaMemcpyHostToDevice, 0));
    CUDA_CHECK(cudaMemcpyAsync(d_c, c, bytes, cudaMemcpyHostToDevice, 0));
    triad<<<(n + 255) / 256, 256>>>(d_a, d_b, d_c, 3.0, n);
    CUDA_CHECK_LAUNCH();
    CUDA_CHECK(cudaMemcpyAsync(a, d_a, bytes, cudaMemcpyDeviceToHost, 0));
    const float ms_serial = timer.stop();

    // 2. pipelined: chunk k goes to stream k % nstreams; work in different streams may overlap
    cudaStream_t streams[nstreams];
    for (int s = 0; s < nstreams; ++s)
        CUDA_CHECK(cudaStreamCreate(&streams[s]));
    timer.start();                              // events in stream 0 wait for all other streams (legacy default stream)
    for (int k = 0; k < nchunks; ++k) {
        cudaStream_t s = streams[k % nstreams];
        const int64_t off = k * chunk;
        CUDA_CHECK(cudaMemcpyAsync(d_b + off, b + off, chunk * sizeof(double), cudaMemcpyHostToDevice, s));
        CUDA_CHECK(cudaMemcpyAsync(d_c + off, c + off, chunk * sizeof(double), cudaMemcpyHostToDevice, s));
        triad<<<(chunk + 255) / 256, 256, 0, s>>>(d_a + off, d_b + off, d_c + off, 3.0, chunk);
        CUDA_CHECK_LAUNCH();
        CUDA_CHECK(cudaMemcpyAsync(a + off, d_a + off, chunk * sizeof(double), cudaMemcpyDeviceToHost, s));
    }
    const float ms_pipelined = timer.stop();

    int64_t wrong = 0;
    for (int64_t i = 0; i < n; ++i)
        wrong += a[i] != 7.0;
    printf("one stream          : %6.2f ms\n", ms_serial);
    printf("%d chunks, %d streams: %6.2f ms  (%lld wrong results)\n", nchunks, nstreams, ms_pipelined, (long long)wrong);

    for (int s = 0; s < nstreams; ++s)
        CUDA_CHECK(cudaStreamDestroy(streams[s]));
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
    CUDA_CHECK(cudaFreeHost(a));
    CUDA_CHECK(cudaFreeHost(b));
    CUDA_CHECK(cudaFreeHost(c));
    return 0;
}
