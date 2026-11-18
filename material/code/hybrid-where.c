#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <mpi.h>
#include <omp.h>
#include <sched.h>
#include <string.h>

int main(int argc, char *argv[]) {
    int rank, size, provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    if (provided < MPI_THREAD_FUNNELED) {
        fprintf(stderr, "MPI_THREAD_FUNNELED is not supported\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    #define MASK_LEN 128

    char host[MPI_MAX_PROCESSOR_NAME] = { 0 };
    int len;
    MPI_Get_processor_name(host, &len);

    // every thread records where it runs and where it may run
    const int nthreads = omp_get_max_threads();
    int  *cpus  = malloc(nthreads * sizeof(int));
    char *masks = calloc(nthreads, MASK_LEN);

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        char *mask = masks + tid * MASK_LEN;
    #ifdef __linux__
        cpus[tid] = sched_getcpu();
        cpu_set_t set;
        sched_getaffinity(0, sizeof(set), &set);       // 0: the calling thread
        int pos = 0;
        for (int c = 0; c < CPU_SETSIZE && pos < MASK_LEN - 8; ++c)
            if (CPU_ISSET(c, &set))
                pos += snprintf(mask + pos, MASK_LEN - pos, "%s%d", pos ? "," : "", c);
    #else
        cpus[tid] = -1;
        strcpy(mask, "n/a");
    #endif
    }

    // collect everything on rank 0 (assumes the same number of threads on all processes)
    char *allhosts = NULL, *allmasks = NULL;
    int  *allcpus  = NULL;
    if (rank == 0) {
        allhosts = malloc(size * MPI_MAX_PROCESSOR_NAME);
        allcpus  = malloc(size * nthreads * sizeof(int));
        allmasks = malloc(size * nthreads * MASK_LEN);
    }
    MPI_Gather(host, MPI_MAX_PROCESSOR_NAME, MPI_CHAR, allhosts, MPI_MAX_PROCESSOR_NAME, MPI_CHAR, 0, MPI_COMM_WORLD);
    MPI_Gather(cpus, nthreads, MPI_INT, allcpus, nthreads, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Gather(masks, nthreads * MASK_LEN, MPI_CHAR, allmasks, nthreads * MASK_LEN, MPI_CHAR, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        const char *places = getenv("OMP_PLACES"), *bind = getenv("OMP_PROC_BIND");
        printf("OMP_PLACES=%s, OMP_PROC_BIND=%s\n", places ? places : "(unset)", bind ? bind : "(unset)");
        for (int r = 0; r < size; ++r)
            for (int t = 0; t < nthreads; ++t) {
                int cpu = allcpus[r * nthreads + t];
                char cpustr[16];
                snprintf(cpustr, sizeof(cpustr), "%d", cpu);
                printf("rank %d on %s, thread %d: on CPU %s, may run on CPUs %s\n", r, allhosts + r * MPI_MAX_PROCESSOR_NAME,
                       t, cpu < 0 ? "n/a" : cpustr, allmasks + (r * nthreads + t) * MASK_LEN);
            }
    }

    free(cpus);
    free(masks);
    free(allhosts);
    free(allcpus);
    free(allmasks);

    MPI_Finalize();
    return 0;
}
