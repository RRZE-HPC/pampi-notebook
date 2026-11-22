#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <mpi.h>
#include <mpicuda.h>

#include <string.h>

// Sends n bytes from GPU memory of rank 0 to GPU memory of rank 1, staged through pinned host memory.

// blocking: copy everything, send everything, copy everything
static void staged(char *d_buf, char *h_buf, size_t n, int rank) {
    if (rank == 0) {
        CUDA_CHECK(cudaMemcpy(h_buf, d_buf, n, cudaMemcpyDeviceToHost));
        MPI_Send(h_buf, (int)n, MPI_BYTE, 1, 0, MPI_COMM_WORLD);
    } else {
        MPI_Recv(h_buf, (int)n, MPI_BYTE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        CUDA_CHECK(cudaMemcpy(d_buf, h_buf, n, cudaMemcpyHostToDevice));
    }
}

// pipelined: in nchunks chunks, so that the copies and the transfers of different chunks overlap
static void pipelined(char *d_buf, char *h_buf, size_t n, int nchunks, cudaStream_t stream, cudaEvent_t *done, int rank) {
    const size_t chunk = n / nchunks;               // n is a multiple of nchunks
    MPI_Request *requests = (MPI_Request *)malloc(nchunks * sizeof(MPI_Request));
    (void)chunk;                                    // silences the compiler warning until you use it
    if (rank == 0) {
        // TODO: enqueue the copies of all chunks to the host in the stream, with an event after each;
        //       wait for the event of chunk k (cudaEventSynchronize - or better: poll it with
        //       cudaEventQuery and call MPI_Testall on the chunks sent so far), then start its
        //       MPI_Isend(); finally, wait for all sends
    } else {
        // TODO: start an MPI_Irecv() for every chunk; wait for the chunks in order
        //       (MPI_Wait) and enqueue the copy of each one to the GPU as soon as it has arrived;
        //       finally, wait for the stream
    }
    free(requests);
}

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    select_gpu();

    const size_t n = (size_t)64 << 20;              // 64 MiB
    const int reps = 10, max_chunks = 64;
    char *d_buf, *h_buf;
    CUDA_CHECK(cudaMalloc(&d_buf, n));
    CUDA_CHECK(cudaMallocHost(&h_buf, n));
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    cudaEvent_t done[max_chunks];                   // done[k]: chunk k has been copied
    for (int k = 0; k < max_chunks; ++k)
        CUDA_CHECK(cudaEventCreateWithFlags(&done[k], cudaEventDisableTiming));

    for (int nchunks = 0; nchunks <= max_chunks; nchunks = nchunks ? 4 * nchunks : 1) {
        // the data to send: byte i is (unsigned char)(i * 7)
        unsigned char *data = (unsigned char *)h_buf;
        for (size_t i = 0; i < n; ++i)
            data[i] = (unsigned char)(rank == 0 ? i * 7 : 0);
        CUDA_CHECK(cudaMemcpy(d_buf, h_buf, n, cudaMemcpyHostToDevice));
        memset(h_buf, 0, n);

        double t = 0.0;
        for (int r = -1; r < reps; ++r) {           // one round to warm up
            MPI_Barrier(MPI_COMM_WORLD);
            const double start = MPI_Wtime();
            if (nchunks == 0)
                staged(d_buf, h_buf, n, rank);
            else
                pipelined(d_buf, h_buf, n, nchunks, stream, done, rank);
            MPI_Barrier(MPI_COMM_WORLD);
            if (r >= 0)
                t += MPI_Wtime() - start;
        }
        t /= reps;

        // check what arrived on the GPU of rank 1
        size_t wrong = 0;
        if (rank == 1) {
            CUDA_CHECK(cudaMemcpy(h_buf, d_buf, n, cudaMemcpyDeviceToHost));
            for (size_t i = 0; i < n; ++i)
                wrong += data[i] != (unsigned char)(i * 7);
        }
        MPI_Bcast(&wrong, sizeof(wrong), MPI_BYTE, 1, MPI_COMM_WORLD);
        if (rank == 0) {
            if (nchunks == 0)
                printf("%-22s", "blocking");
            else
                printf("pipelined, %2d chunks  ", nchunks);
            printf(": %7.2f ms, %5.1f GB/s, %zu wrong bytes\n", 1e3 * t, n / t / 1e9, wrong);
        }
    }

    for (int k = 0; k < max_chunks; ++k)
        CUDA_CHECK(cudaEventDestroy(done[k]));
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(d_buf));
    CUDA_CHECK(cudaFreeHost(h_buf));
    MPI_Finalize();
    return 0;
}
