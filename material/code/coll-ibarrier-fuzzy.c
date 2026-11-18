#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <mpi.h>

int main(int argc, char *argv[]) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    double t0 = MPI_Wtime();
    while (MPI_Wtime() - t0 < 0.1 * rank)
        ;                                           // phase 1: rank r is busy for 0.1 * r seconds

    double t1 = MPI_Wtime();
    MPI_Request request;
    MPI_Ibarrier(MPI_COMM_WORLD, &request);         // "I have finished phase 1"

    // independent work (e.g. postprocessing, writing output) in small chunks until all ranks have arrived
    long chunks = 0;
    int flag = 0;
    MPI_Test(&request, &flag, MPI_STATUS_IGNORE);
    while (!flag) {
        double t = MPI_Wtime();
        while (MPI_Wtime() - t < 1e-3)
            ;                                       // 1 ms of independent work
        ++chunks;
        MPI_Test(&request, &flag, MPI_STATUS_IGNORE);
    }
    double t2 = MPI_Wtime();

    printf("rank %d: arrived after %.2f s, did %3ld ms of independent work in the %.2f s until all had arrived\n",
           rank, t1 - t0, chunks, t2 - t1);

    MPI_Finalize();
    return 0;
}
