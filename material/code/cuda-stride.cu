#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <cuda-check.h>

// thread t copies element t * stride
__global__ void strided_copy(float *out, const float *in, int stride, int64_t n) {
    int64_t i = (blockIdx.x * (int64_t)blockDim.x + threadIdx.x) * stride;
    if (i < n)
        out[i] = in[i];
}

int main(void) {
    const int64_t n = 1 << 26;                  // 256 MiB per array
    float *in, *out;
    CUDA_CHECK(cudaMalloc(&in, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&out, n * sizeof(float)));
    CUDA_CHECK(cudaMemset(in, 0, n * sizeof(float)));

    GpuTimer timer;
    strided_copy<<<n / 256, 256>>>(out, in, 1, n);         // warm-up
    CUDA_CHECK_LAUNCH();
    printf("stride   elements  time [ms]  useful GB/s\n");
    for (int stride = 1; stride <= 64; stride *= 2) {
        const int64_t count = n / stride;        // elements copied
        timer.start();
        strided_copy<<<(count + 255) / 256, 256>>>(out, in, stride, n);
        float ms = timer.stop();
        CUDA_CHECK_LAUNCH();
        printf("%6d  %9lld  %9.3f  %11.1f\n", stride, (long long)count, ms, 2 * count * sizeof(float) / ms / 1e6);
    }

    CUDA_CHECK(cudaFree(in));
    CUDA_CHECK(cudaFree(out));
    return 0;
}
