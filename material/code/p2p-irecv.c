#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <mpi.h>

int main(int argc, char *argv[]) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int value = -1;

    if (rank == 0) {
        double t0 = MPI_Wtime();
        while (MPI_Wtime() - t0 < 0.2)
            ;                                       // busy for 0.2 s
        value = 42;
        MPI_Send(&value, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);
    } else if (rank == 1) {
        MPI_Request request;
        MPI_Irecv(&value, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &request);

        // erroneous: the buffer must not be accessed before the receive has completed!
        printf("rank 1: after MPI_Irecv(): value = %d\n", value);

        MPI_Wait(&request, MPI_STATUS_IGNORE);
        printf("rank 1: after MPI_Wait():  value = %d\n", value);
    }

    MPI_Finalize();
    return 0;
}
