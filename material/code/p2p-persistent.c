#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <mpi.h>

int main(int argc, char *argv[]) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int left  = (rank - 1 + size) % size;
    int right = (rank + 1) % size;
    int sendval = 0, recvval = -1;

    // set up the ring shift once ...
    MPI_Request requests[2];
    MPI_Recv_init(&recvval, 1, MPI_INT, left,  0, MPI_COMM_WORLD, &requests[0]);
    MPI_Send_init(&sendval, 1, MPI_INT, right, 0, MPI_COMM_WORLD, &requests[1]);

    // ... and start it in every iteration
    for (int iter = 0; iter < 3; ++iter) {
        sendval = 100 * iter + rank;            // the buffer is fixed, its contents may change
        MPI_Startall(2, requests);
        MPI_Waitall(2, requests, MPI_STATUSES_IGNORE);
    }

    printf("rank %d: received %d in the last iteration\n", rank, recvval);

    MPI_Request_free(&requests[0]);
    MPI_Request_free(&requests[1]);

    MPI_Finalize();
    return 0;
}
