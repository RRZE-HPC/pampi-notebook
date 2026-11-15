#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <mpi.h>

int main(int argc, char *argv[]) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size != 3) {
        if (rank == 0)
            fprintf(stderr, "This example requires exactly 3 processes, got %d\n", size);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // how many elements rank r sends to rank j: counts[r][j] (the example of the figure)
    const int counts[3][3] = { { 1, 2, 1 },
                               { 0, 1, 2 },
                               { 3, 1, 1 } };

    // send side: every rank knows how much it sends to whom
    int sendcounts[3], sdispls[3], nsend = 0;
    for (int j = 0; j < size; ++j) {
        sendcounts[j] = counts[rank][j];
        sdispls[j]    = nsend;                      // the chunks are stored back to back
        nsend        += sendcounts[j];
    }
    int *sendbuf = malloc(nsend * sizeof(int));
    for (int k = 0; k < nsend; ++k)
        sendbuf[k] = 10 * rank + k;

    // receive side: tell every rank how much it will get from us
    int recvcounts[3], rdispls[3], nrecv = 0;
    MPI_Alltoall(sendcounts, 1, MPI_INT, recvcounts, 1, MPI_INT, MPI_COMM_WORLD);
    for (int i = 0; i < size; ++i) {
        rdispls[i] = nrecv;
        nrecv     += recvcounts[i];
    }
    int *recvbuf = malloc(nrecv * sizeof(int));

    MPI_Alltoallv(sendbuf, sendcounts, sdispls, MPI_INT,
                  recvbuf, recvcounts, rdispls, MPI_INT, MPI_COMM_WORLD);

    char line[256];
    int pos = snprintf(line, sizeof(line), "rank %d: recvcounts = {%d, %d, %d}  recvbuf =",
                       rank, recvcounts[0], recvcounts[1], recvcounts[2]);
    for (int k = 0; k < nrecv; ++k)
        pos += snprintf(line + pos, sizeof(line) - pos, " %d", recvbuf[k]);
    printf("%s\n", line);

    free(sendbuf);
    free(recvbuf);

    MPI_Finalize();
    return 0;
}
