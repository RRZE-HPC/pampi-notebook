#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <mpi.h>

int main(int argc, char *argv[]) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    double data[5];

    if (rank == 0) {
        for (int i = 0; i < 5; ++i)
            data[i] = 1.5 * i;

        MPI_Send(data, 5, MPI_DOUBLE, 1, 42, MPI_COMM_WORLD);

        // the send buffer may safely be reused as soon as MPI_Send() returns
        data[0] = -1.0;
        printf("rank 0: sent data\n");
    } else if (rank == 1) {
        MPI_Status status;
        MPI_Recv(data, 5, MPI_DOUBLE, 0, 42, MPI_COMM_WORLD, &status);

        printf("rank 1: received");
        for (int i = 0; i < 5; ++i)
            printf(" %.1f", data[i]);
        printf("\n");
    }

    MPI_Finalize();
    return 0;
}
