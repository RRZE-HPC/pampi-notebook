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

    for (int n = 1; n <= (1 << 20); n *= 16) {
        int *sendbuf = malloc(n * sizeof(int));
        int *recvbuf = malloc(n * sizeof(int));
        for (int i = 0; i < n; ++i)
            sendbuf[i] = rank;

        MPI_Send(sendbuf, n, MPI_INT, right, 0, MPI_COMM_WORLD);
        MPI_Recv(recvbuf, n, MPI_INT, left,  0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        if (rank == 0)
            printf("%9zu bytes: rank 0 received data from rank %d\n", n * sizeof(int), recvbuf[0]);

        free(sendbuf);
        free(recvbuf);
    }

    MPI_Finalize();
    return 0;
}
