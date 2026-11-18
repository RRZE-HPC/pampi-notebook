#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <mpi.h>

int main(int argc, char *argv[]) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int *sendbuf = malloc(size * sizeof(int));
    int *recvbuf = malloc(size * sizeof(int));
    for (int i = 0; i < size; ++i)
        sendbuf[i] = rank * size + i;               // row `rank` of a size x size matrix

    MPI_Alltoall(sendbuf, 1, MPI_INT, recvbuf, 1, MPI_INT, MPI_COMM_WORLD);

    char line[256];
    int pos = snprintf(line, sizeof(line), "rank %d: sent", rank);
    for (int i = 0; i < size; ++i)
        pos += snprintf(line + pos, sizeof(line) - pos, " %2d", sendbuf[i]);
    pos += snprintf(line + pos, sizeof(line) - pos, "   received");
    for (int i = 0; i < size; ++i)
        pos += snprintf(line + pos, sizeof(line) - pos, " %2d", recvbuf[i]);
    printf("%s\n", line);

    free(sendbuf);
    free(recvbuf);

    MPI_Finalize();
    return 0;
}
