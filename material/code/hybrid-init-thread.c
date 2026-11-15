#include <stdio.h>
#include <mpi.h>
#include <omp.h>

const char *level_name(int level) {
    return level == MPI_THREAD_SINGLE     ? "MPI_THREAD_SINGLE"
         : level == MPI_THREAD_FUNNELED   ? "MPI_THREAD_FUNNELED"
         : level == MPI_THREAD_SERIALIZED ? "MPI_THREAD_SERIALIZED"
         :                                  "MPI_THREAD_MULTIPLE";
}

int main(int argc, char **argv) {
    int provided, rank;

    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int queried;
    MPI_Query_thread(&queried);                     // the provided level, available at any time
    if (rank == 0)
        printf("required MPI_THREAD_MULTIPLE, provided %s (MPI_Query_thread: %s)\n", level_name(provided), level_name(queried));

    #pragma omp parallel
    {
        int is_main;
        MPI_Is_thread_main(&is_main);               // allowed from any thread
        printf("rank %d, thread %d: main thread? %s\n", rank, omp_get_thread_num(), is_main ? "yes" : "no");
    }

    MPI_Finalize();
    return 0;
}
