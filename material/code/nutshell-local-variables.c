#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <mpi.h>

int main(int argc, char *argv[]) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int x = 42;

    if (rank == 0)
        x = 0;      // only changes the copy of rank 0

    printf("rank %d: x = %d\n", rank, x);

    MPI_Finalize();
    return 0;
}
