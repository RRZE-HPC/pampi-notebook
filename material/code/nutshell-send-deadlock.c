#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <mpi.h>

int main(int argc, char *argv[]) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int other = 1 - rank;   // the partner process (run with exactly 2 processes)

    for (int n = 1; n <= (1 << 24); n *= 4) {
        char *sendbuf = calloc(n, 1);
        char *recvbuf = malloc(n);

        MPI_Send(sendbuf, n, MPI_BYTE, other, 0, MPI_COMM_WORLD);
        MPI_Recv(recvbuf, n, MPI_BYTE, other, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        if (rank == 0)
            printf("%8d bytes exchanged\n", n);

        free(sendbuf);
        free(recvbuf);
    }

    MPI_Finalize();
    return 0;
}
