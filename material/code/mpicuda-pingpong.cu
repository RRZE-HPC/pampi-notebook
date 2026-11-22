#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <mpi.h>
#include <mpicuda.h>

#include <string.h>

// the datapaths: the data lives in GPU memory on both sides, except for the host baseline
enum { HOST, PAGEABLE, PINNED, MANAGED, DEVICE, NPATHS };
static const char *names[NPATHS] = { "host", "pageable", "pinned", "managed", "device" };

struct Buffers {
    char *dev;                                      // cudaMalloc(): the data on the GPU
    char *host;                                     // malloc(): pageable host memory
    char *pinned;                                   // cudaMallocHost(): pinned host memory
    char *managed;                                  // cudaMallocManaged(): migrates between CPU and GPU
};

// writes one byte of every page on the GPU - like a kernel that uses the received data,
// it moves the pages of a managed buffer to the GPU
__global__ void touch(char *buf, size_t bytes) {
    const size_t stride = (size_t)gridDim.x * blockDim.x * 4096;
    for (size_t i = (blockIdx.x * (size_t)blockDim.x + threadIdx.x) * 4096; i < bytes; i += stride)
        buf[i] += 1;
}

// sends a message whose data was produced on the GPU
static void send_msg(int path, Buffers *b, size_t bytes, int peer) {
    switch (path) {
    case HOST:                                      // baseline: no GPU involved
        MPI_Send(b->host, (int)bytes, MPI_BYTE, peer, 0, MPI_COMM_WORLD);
        break;
    case PAGEABLE:                                  // staged: GPU -> host, then MPI
    case PINNED: {
        char *staging = path == PINNED ? b->pinned : b->host;
        CUDA_CHECK(cudaMemcpy(staging, b->dev, bytes, cudaMemcpyDeviceToHost));
        MPI_Send(staging, (int)bytes, MPI_BYTE, peer, 0, MPI_COMM_WORLD);
        break;
    }
    case MANAGED:                                   // read on the CPU (or by a copy engine, if CUDA-aware)
        MPI_Send(b->managed, (int)bytes, MPI_BYTE, peer, 0, MPI_COMM_WORLD);
        break;
    case DEVICE:                                    // CUDA-aware MPI: the device pointer itself
        MPI_Send(b->dev, (int)bytes, MPI_BYTE, peer, 0, MPI_COMM_WORLD);
        break;
    }
}

// receives a message whose data is needed on the GPU
static void recv_msg(int path, Buffers *b, size_t bytes, int peer) {
    switch (path) {
    case HOST:
        MPI_Recv(b->host, (int)bytes, MPI_BYTE, peer, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        break;
    case PAGEABLE:                                  // staged: MPI, then host -> GPU
    case PINNED: {
        char *staging = path == PINNED ? b->pinned : b->host;
        MPI_Recv(staging, (int)bytes, MPI_BYTE, peer, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        CUDA_CHECK(cudaMemcpy(b->dev, staging, bytes, cudaMemcpyHostToDevice));
        break;
    }
    case MANAGED:                                   // written by MPI, then the GPU uses the data
        MPI_Recv(b->managed, (int)bytes, MPI_BYTE, peer, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        touch<<<64, 256>>>(b->managed, bytes);
        CUDA_CHECK_LAUNCH();
        CUDA_CHECK(cudaDeviceSynchronize());
        break;
    case DEVICE:
        MPI_Recv(b->dev, (int)bytes, MPI_BYTE, peer, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        break;
    }
}

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (size != 2) {
        if (rank == 0)
            fprintf(stderr, "the ping-pong needs exactly 2 processes\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    const int gpu = select_gpu(), peer = 1 - rank;

    const char *how;
    const int aware = mpi_cuda_aware(&how);

    // where are the two processes and their GPUs?
    char host[MPI_MAX_PROCESSOR_NAME] = { 0 }, peer_host[MPI_MAX_PROCESSOR_NAME] = { 0 };
    int len, peer_gpu;
    MPI_Get_processor_name(host, &len);
    MPI_Sendrecv(host, MPI_MAX_PROCESSOR_NAME, MPI_CHAR, peer, 0, peer_host, MPI_MAX_PROCESSOR_NAME, MPI_CHAR, peer, 0,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Sendrecv(&gpu, 1, MPI_INT, peer, 0, &peer_gpu, 1, MPI_INT, peer, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    if (rank == 0) {
        if (strcmp(host, peer_host) != 0)
            printf("rank 0: GPU %d on %s, rank 1: GPU %d on %s - different nodes\n", gpu, host, peer_gpu, peer_host);
        else if (gpu == peer_gpu)
            printf("both ranks on %s share GPU %d\n", host, gpu);
        else {
            int p2p = 0;                            // can the two GPUs access each other's memory?
            CUDA_CHECK(cudaDeviceCanAccessPeer(&p2p, gpu, peer_gpu));
            printf("rank 0: GPU %d, rank 1: GPU %d on %s, peer access: %s\n", gpu, peer_gpu, host, p2p ? "yes" : "no");
        }
        printf("CUDA-aware MPI: %s (%s)%s\n\n", aware == 1 ? "yes" : aware == 0 ? "no" : "unknown", how,
               aware == 1 ? "" : " - the device path is skipped");
        printf("%10s", "bytes");
        for (int p = 0; p < NPATHS; ++p)
            printf(" %10s", names[p]);
        printf("   [time per message in us]\n");
    }

    const size_t max_bytes = (size_t)32 << 20;     // 32 MiB
    Buffers b;
    CUDA_CHECK(cudaMalloc(&b.dev, max_bytes));
    CUDA_CHECK(cudaMallocHost(&b.pinned, max_bytes));
    CUDA_CHECK(cudaMallocManaged(&b.managed, max_bytes));
    b.host = (char *)malloc(max_bytes);
    CUDA_CHECK(cudaMemset(b.dev, rank, max_bytes));
    memset(b.host, rank, max_bytes);
    memset(b.pinned, rank, max_bytes);
    memset(b.managed, rank, max_bytes);

    for (size_t bytes = 8; bytes <= max_bytes; bytes *= 4) {
        const size_t n = ((size_t)1 << 28) / bytes;                    // fewer repetitions for long messages
        const int reps = n > 1000 ? 1000 : n < 10 ? 10 : (int)n;
        if (rank == 0)
            printf("%10zu", bytes);
        for (int p = 0; p < NPATHS; ++p) {
            if (p == DEVICE && aware != 1) {
                if (rank == 0)
                    printf(" %10s", "n/a");
                continue;
            }
            if (p == MANAGED) {                     // the data starts on the GPU, as for the other paths
                touch<<<64, 256>>>(b.managed, bytes);
                CUDA_CHECK_LAUNCH();
                CUDA_CHECK(cudaDeviceSynchronize());
            }
            double start = 0.0;
            // 2 round trips to warm up: connections, registration of the buffers, CUDA IPC handles
            for (int r = -2; r < reps; ++r) {
                if (r == 0) {
                    MPI_Barrier(MPI_COMM_WORLD);
                    start = MPI_Wtime();
                }
                if (rank == 0) {
                    send_msg(p, &b, bytes, peer);
                    recv_msg(p, &b, bytes, peer);
                } else {
                    recv_msg(p, &b, bytes, peer);
                    send_msg(p, &b, bytes, peer);
                }
            }
            const double t = (MPI_Wtime() - start) / (2.0 * reps);      // one way: half a round trip
            if (rank == 0)
                printf(" %10.2f", 1e6 * t);
        }
        if (rank == 0) {
            printf("\n");
            fflush(stdout);
        }
    }

    CUDA_CHECK(cudaFree(b.dev));
    CUDA_CHECK(cudaFreeHost(b.pinned));
    CUDA_CHECK(cudaFree(b.managed));
    free(b.host);
    MPI_Finalize();
    return 0;
}
