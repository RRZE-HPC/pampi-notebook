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
        ;                                           // rank r is busy for 0.1 * r seconds

    double t1 = MPI_Wtime();
    MPI_Barrier(MPI_COMM_WORLD);
    double t2 = MPI_Wtime();

    printf("rank %d: arrived after %.2f s, waited %.2f s in the barrier\n", rank, t1 - t0, t2 - t1);

    MPI_Finalize();
    return 0;
}
