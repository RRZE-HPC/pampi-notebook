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

    const int n = 1 << 20;      // 4 MiB, far above the eager limit
    int *sendbuf = malloc(n * sizeof(int));
    int *recvbuf = malloc(n * sizeof(int));
    for (int i = 0; i < n; ++i)
        sendbuf[i] = rank;

    MPI_Request requests[2];
    MPI_Irecv(recvbuf, n, MPI_INT, left,  0, MPI_COMM_WORLD, &requests[0]);
    MPI_Isend(sendbuf, n, MPI_INT, right, 0, MPI_COMM_WORLD, &requests[1]);

    // work that does not touch sendbuf or recvbuf could be done here

    MPI_Waitall(2, requests, MPI_STATUSES_IGNORE);
    printf("rank %d: received data from rank %d\n", rank, recvbuf[0]);

    free(sendbuf);
    free(recvbuf);

    MPI_Finalize();
    return 0;
}
