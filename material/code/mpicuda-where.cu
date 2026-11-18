#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <mpi.h>
#include <mpicuda.h>

#ifdef __linux__
#include <sched.h>                                  // sched_getcpu()
#endif

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const int local = node_rank();
    const int gpu = select_gpu();                   // first CUDA call apart from counting the GPUs

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, gpu));
    char bus[32];                                   // the PCI address identifies the GPU in nvidia-smi
    CUDA_CHECK(cudaDeviceGetPCIBusId(bus, sizeof(bus), gpu));
    int ngpus;
    CUDA_CHECK(cudaGetDeviceCount(&ngpus));

    char host[MPI_MAX_PROCESSOR_NAME];
    int len;
    MPI_Get_processor_name(host, &len);
    const char *visible = getenv("CUDA_VISIBLE_DEVICES");
#ifdef __linux__
    const int cpu = sched_getcpu();
#else
    const int cpu = -1;
#endif

    // one line per process, collected on rank 0 and printed in order
    char line[256];
    snprintf(line, sizeof(line), "rank %d on %s (node rank %d, CPU %d): GPU %d of %d [%s], %s, CUDA_VISIBLE_DEVICES=%s",
             rank, host, local, cpu, gpu, ngpus, bus, prop.name, visible ? visible : "(unset)");
    char *lines = rank == 0 ? (char *)malloc(size * sizeof(line)) : NULL;
    MPI_Gather(line, sizeof(line), MPI_CHAR, lines, sizeof(line), MPI_CHAR, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        for (int r = 0; r < size; ++r)
            printf("%s\n", lines + r * sizeof(line));
        free(lines);

        const char *how;
        const int aware = mpi_cuda_aware(&how);
        char kinds[MPI_MAX_INFO_VAL + 1];
        mpi_memory_kinds(kinds, sizeof(kinds));
        printf("\nCUDA-aware MPI : %s (%s)\n", aware == 1 ? "yes" : aware == 0 ? "no" : "unknown", how);
        printf("memory kinds   : %s\n", kinds);
    }

    MPI_Finalize();
    return 0;
}
