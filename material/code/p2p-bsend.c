#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <mpi.h>

int main(int argc, char *argv[]) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int other = 1 - rank;       // the partner process (run with exactly 2 processes)
    const int n = 1 << 20;      // 1 Mi doubles = 8 MiB, far above the eager limit

    double *sendbuf = calloc(n, sizeof(double));
    double *recvbuf = malloc(n * sizeof(double));

    // attach a buffer that is large enough for one message
    int bufsize;
    MPI_Pack_size(n, MPI_DOUBLE, MPI_COMM_WORLD, &bufsize);
    bufsize += MPI_BSEND_OVERHEAD;
    void *buffer = malloc(bufsize);
    MPI_Buffer_attach(buffer, bufsize);

    MPI_Bsend(sendbuf, n, MPI_DOUBLE, other, 0, MPI_COMM_WORLD);
    MPI_Recv(recvbuf, n, MPI_DOUBLE, other, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    printf("rank %d: exchanged %zu bytes\n", rank, n * sizeof(double));

    // waits until the buffered message has been delivered
    MPI_Buffer_detach(&buffer, &bufsize);
    free(buffer);
    free(sendbuf);
    free(recvbuf);

    MPI_Finalize();
    return 0;
}
