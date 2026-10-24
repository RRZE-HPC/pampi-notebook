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

    MPI_Sendrecv(sendbuf, n, MPI_INT, right, 0,
                 recvbuf, n, MPI_INT, left,  0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    printf("rank %d: received data from rank %d\n", rank, recvbuf[0]);

    free(sendbuf);
    free(recvbuf);

    MPI_Finalize();
    return 0;
}
