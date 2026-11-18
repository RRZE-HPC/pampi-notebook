#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <mpi.h>

int main(int argc, char *argv[]) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const int root = 1;
    int buffer[3] = { 0, 0, 0 };

    if (rank == root) {
        buffer[0] = 1;
        buffer[1] = 2;
        buffer[2] = 3;
    }

    MPI_Bcast(buffer, 3, MPI_INT, root, MPI_COMM_WORLD);

    printf("rank %d: buffer = { %d, %d, %d }\n", rank, buffer[0], buffer[1], buffer[2]);

    MPI_Finalize();
    return 0;
}
