#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <cuda-check.h>

// TODO: every thread sums 4 / (1 + x^2) at the midpoints x = (i + 0.5) * h of its share of the
//       n intervals (grid-stride loop); combine the partial sums of the block as in the
//       reduction example and add the result of the block to *sum with atomicAdd()
__global__ void pi_kernel(int64_t n, double h, double *sum) {
}

int main(void) {
    const int64_t n = 1000000000;
    const double h = 1.0 / n;

    double *d_sum, sum;
    CUDA_CHECK(cudaMalloc(&d_sum, sizeof(double)));
    CUDA_CHECK(cudaMemset(d_sum, 0, sizeof(double)));

    GpuTimer timer;
    timer.start();
    pi_kernel<<<4 * sm_count(), 256>>>(n, h, d_sum);
    float ms = timer.stop();
    CUDA_CHECK_LAUNCH();
    CUDA_CHECK(cudaMemcpy(&sum, d_sum, sizeof(double), cudaMemcpyDeviceToHost));
    printf("pi = %.15f, error %.2e, %.2f ms\n", sum * h, sum * h - 3.14159265358979323846, ms);

    CUDA_CHECK(cudaFree(d_sum));
    return 0;
}
