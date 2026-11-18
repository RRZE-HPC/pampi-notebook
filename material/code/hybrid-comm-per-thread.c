#include <stdio.h>
#include <mpi.h>
#include <omp.h>

static void busy_wait(double seconds) {
    double t0 = omp_get_wtime();
    while (omp_get_wtime() - t0 < seconds)
        ;
}

int main(int argc, char **argv) {
    int provided, rank;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (provided < MPI_THREAD_MULTIPLE) {
        if (rank == 0)
            fprintf(stderr, "requires MPI_THREAD_MULTIPLE\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // one communicator per thread, created before the parallel region
    MPI_Comm comms[2];
    for (int t = 0; t < 2; ++t)
        MPI_Comm_dup(MPI_COMM_WORLD, &comms[t]);

    int a = rank == 0 ? 1 : 0, b = rank == 0 ? 2 : 0;
    #pragma omp parallel num_threads(2)
    {
        int tid = omp_get_thread_num();
        if ((rank == 0 && tid == 1) || (rank == 1 && tid == 0))
            busy_wait(0.2);

        if (tid == 0)
            MPI_Bcast(&a, 1, MPI_INT, 0, comms[0]);   // each thread on its own communicator
        else
            MPI_Bcast(&b, 1, MPI_INT, 0, comms[1]);
    }
    printf("rank %d: a = %d, b = %d\n", rank, a, b);

    for (int t = 0; t < 2; ++t)
        MPI_Comm_free(&comms[t]);
    MPI_Finalize();
    return 0;
}
