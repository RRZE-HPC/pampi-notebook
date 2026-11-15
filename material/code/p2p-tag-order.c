#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <mpi.h>

int main(int argc, char *argv[]) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        int a = 1, b = 2;
        MPI_Send(&a, 1, MPI_INT, 1, 1, MPI_COMM_WORLD);     // tag 1 first ...
        MPI_Send(&b, 1, MPI_INT, 1, 2, MPI_COMM_WORLD);     // ... then tag 2
    } else if (rank == 1) {
        int a, b;
        MPI_Recv(&b, 1, MPI_INT, 0, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);   // tag 2 first!
        MPI_Recv(&a, 1, MPI_INT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf("rank 1: a = %d (tag 1), b = %d (tag 2)\n", a, b);
    }

    MPI_Finalize();
    return 0;
}
