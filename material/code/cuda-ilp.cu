#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <cuda-check.h>

// copy with ILP independent loads per thread and loop iteration: all ILP loads are issued
// before the first result is needed - like the independent loads of an out-of-order CPU core
template <int ILP>
__global__ void copy(float *out, const float *in, int64_t n) {
    const int64_t stride = (int64_t)gridDim.x * blockDim.x;
    for (int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; i < n; i += ILP * stride) {
        float v[ILP];
        #pragma unroll
        for (int k = 0; k < ILP; ++k)
            if (i + k * stride < n)
                v[k] = in[i + k * stride];
        #pragma unroll
        for (int k = 0; k < ILP; ++k)
            if (i + k * stride < n)
                out[i + k * stride] = v[k];
    }
}

// bandwidth in GB/s of the copy with a grid of w warps per SM
template <int ILP>
double bandwidth(float *out, const float *in, int64_t n, int sms, int w) {
    const int threads = w == 1 ? 32 : 64;
    const int blocks = sms * w * 32 / threads;
    GpuTimer timer;
    timer.start();
    copy<ILP><<<blocks, threads>>>(out, in, n);
    const float ms = timer.stop();
    CUDA_CHECK_LAUNCH();
    return 2.0 * n * sizeof(float) / ms / 1e6;
}

int main(void) {
    const int64_t n = 1 << 26;                  // 256 MiB per array
    int device, threadsSM;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaDeviceGetAttribute(&threadsSM, cudaDevAttrMaxThreadsPerMultiProcessor, device));
    const int sms = sm_count(), max_warps = threadsSM / 32;

    float *in, *out;
    CUDA_CHECK(cudaMalloc(&in, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&out, n * sizeof(float)));
    CUDA_CHECK(cudaMemset(in, 0, n * sizeof(float)));
    bandwidth<1>(out, in, n, sms, max_warps);   // warm-up

    printf("warps/SM   GB/s with ILP = 1   ILP = 4   ILP = 8\n");
    for (int w = 1; w <= max_warps; w *= 2)
        printf("%8d  %18.0f  %8.0f  %8.0f\n", w, bandwidth<1>(out, in, n, sms, w),
               bandwidth<4>(out, in, n, sms, w), bandwidth<8>(out, in, n, sms, w));

    CUDA_CHECK(cudaFree(in));
    CUDA_CHECK(cudaFree(out));
    return 0;
}
