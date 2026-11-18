#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <mpi.h>
#include <mpicuda.h>

// the GPU's clock in nanoseconds: a special register, read with inline PTX assembly
__device__ unsigned long long globaltimer(void) {
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}

// a slow producer: computes for 100 ms, then writes its result
__global__ void produce(int *x, int n, int value) {
    const unsigned long long start = globaltimer();
    while (globaltimer() - start < 100000000ull)
        ;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        x[i] = value;
}

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    select_gpu();
    const char *how;
    const bool aware = mpi_cuda_aware(&how) == 1;   // otherwise: stage through host memory

    const int n = 1024;
    int *d_x, *h_x = (int *)malloc(n * sizeof(int));
    CUDA_CHECK(cudaMalloc(&d_x, n * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_x, 0, n * sizeof(int)));
    CUDA_CHECK(cudaDeviceSynchronize());

    // a stream of the application, e.g. to overlap kernels; non-blocking: it does not
    // synchronize with the default stream, which cudaMemcpy() uses
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    // warm up: on the first use of a buffer, the library sets up its path (registration, CUDA IPC
    // handles) - which may synchronize the whole GPU and would hide the bug in the first round
    if (rank == 0)
        MPI_Send(aware ? (void *)d_x : (void *)h_x, n, MPI_INT, 1, 0, MPI_COMM_WORLD);
    else if (rank == 1)
        MPI_Recv(aware ? (void *)d_x : (void *)h_x, n, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    if (rank == 0)
        printf("%s\n", aware ? "CUDA-aware MPI: sending the device buffer" : "not CUDA-aware: staging through host memory");
    for (int fixed = 0; fixed <= 1; ++fixed) {
        const int value = 42 + fixed;
        if (rank == 0) {
            produce<<<n / 256, 256, 0, stream>>>(d_x, n, value);
            CUDA_CHECK_LAUNCH();
            if (fixed)
                CUDA_CHECK(cudaStreamSynchronize(stream));          // wait until the data is there
            if (aware)
                MPI_Send(d_x, n, MPI_INT, 1, 0, MPI_COMM_WORLD);
            else {
                CUDA_CHECK(cudaMemcpy(h_x, d_x, n * sizeof(int), cudaMemcpyDeviceToHost));
                MPI_Send(h_x, n, MPI_INT, 1, 0, MPI_COMM_WORLD);
            }
            CUDA_CHECK(cudaStreamSynchronize(stream));              // before the buffer is reused
        } else if (rank == 1) {
            if (aware)
                MPI_Recv(d_x, n, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            else {
                MPI_Recv(h_x, n, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                CUDA_CHECK(cudaMemcpy(d_x, h_x, n * sizeof(int), cudaMemcpyHostToDevice));
            }
            // check on the host what arrived in GPU memory
            CUDA_CHECK(cudaMemcpy(h_x, d_x, n * sizeof(int), cudaMemcpyDeviceToHost));
            int correct = 0;
            for (int i = 0; i < n; ++i)
                correct += h_x[i] == value;
            printf("%-28s: %4d of %d values correct (x[0] = %d, expected %d)\n",
                   fixed ? "with cudaStreamSynchronize()" : "without synchronization", correct, n, h_x[0], value);
        }
    }

    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(d_x));
    free(h_x);
    MPI_Finalize();
    return 0;
}
