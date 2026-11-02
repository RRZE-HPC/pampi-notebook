#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <mpi.h>

int main(int argc, char *argv[]) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int value = rank + 1;
    int inclusive, exclusive;

    MPI_Scan  (&value, &inclusive, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    MPI_Exscan(&value, &exclusive, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    if (rank == 0)
        exclusive = 0;                              // undefined on rank 0

    printf("rank %d: value %d, MPI_Scan %2d, MPI_Exscan %2d\n", rank, value, inclusive, exclusive);

    MPI_Finalize();
    return 0;
}
