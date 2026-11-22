#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <cuda-check.h>

// two code paths with different instructions, so that the compiler cannot merge them
__device__ float path_a(float x, int k) {
    for (int i = 0; i < k; ++i)
        x = x * 0.999f + 0.5f;
    return x;
}
__device__ float path_b(float x, int k) {
    for (int i = 0; i < k; ++i)
        x = sqrtf(x + 1.0f);
    return x;
}

// mode 0: all threads take path A, mode 1: all take path B,
// mode 2: whole warps alternate between A and B, mode 3: neighboring threads alternate
__global__ void branches(float *x, int mode, int k) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    bool a = mode == 0 ? true
           : mode == 1 ? false
           : mode == 2 ? (i / 32) % 2 == 0
           :             i % 2 == 0;
    if (a)
        x[i] = path_a(x[i], k);
    else
        x[i] = path_b(x[i], k);
}

int main(void) {
    const int n = 1 << 22, k = 2000;
    float *x = (float *)malloc(n * sizeof(float)), *d_x;
    for (int i = 0; i < n; ++i)
        x[i] = 1.0f;
    CUDA_CHECK(cudaMalloc(&d_x, n * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_x, x, n * sizeof(float), cudaMemcpyHostToDevice));

    const char *name[] = { "all threads path A", "all threads path B",
                           "warps alternate A/B", "threads alternate A/B" };
    GpuTimer timer;
    branches<<<n / 256, 256>>>(d_x, 0, k);      // warm-up
    CUDA_CHECK_LAUNCH();
    for (int mode = 0; mode < 4; ++mode) {
        timer.start();
        branches<<<n / 256, 256>>>(d_x, mode, k);
        float ms = timer.stop();
        CUDA_CHECK_LAUNCH();
        printf("%-22s: %6.2f ms\n", name[mode], ms);
    }

    CUDA_CHECK(cudaFree(d_x));
    free(x);
    return 0;
}
