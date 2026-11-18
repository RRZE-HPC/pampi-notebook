#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <mpi.h>

int main(int argc, char *argv[]) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size != 4) {
        if (rank == 0)
            fprintf(stderr, "This example requires exactly 4 processes, got %d\n", size);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    const int root = 1;
    int sendbuf[7]    = { 1, 2, 3, 4, 5, 6, 7 };   // only significant at the root
    int sendcounts[4] = { 2, 1, 3, 1 };            // only significant at the root
    int displs[4]     = { 5, 4, 1, 0 };            // only significant at the root

    int recvcount = sendcounts[rank];               // every rank must know the size of its chunk
    int recvbuf[3];

    MPI_Scatterv(sendbuf, sendcounts, displs, MPI_INT, recvbuf, recvcount, MPI_INT, root, MPI_COMM_WORLD);

    printf("rank %d: received", rank);
    for (int i = 0; i < recvcount; ++i)
        printf(" %d", recvbuf[i]);
    printf("\n");

    MPI_Finalize();
    return 0;
}
