#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <mpi.h>

int main(int argc, char *argv[]) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    enum { NBLOCKS = 4, BLOCK = 1000 };
    const int root = 0;
    double buf[2][BLOCK];                           // double buffering: one block is processed, the next one arrives
    int cur = 0;
    double total = 0.0;

    // the root "reads" block b from a file — here, every element of block b is b + 1
    if (rank == root)
        for (int i = 0; i < BLOCK; ++i)
            buf[cur][i] = 1.0;
    MPI_Bcast(buf[cur], BLOCK, MPI_DOUBLE, root, MPI_COMM_WORLD);   // the first block has to be waited for

    for (int b = 0; b < NBLOCKS; ++b) {
        int next = 1 - cur;
        MPI_Request request = MPI_REQUEST_NULL;

        if (b + 1 < NBLOCKS) {                      // start the transfer of the next block
            if (rank == root)
                for (int i = 0; i < BLOCK; ++i)
                    buf[next][i] = b + 2.0;
            MPI_Ibcast(buf[next], BLOCK, MPI_DOUBLE, root, MPI_COMM_WORLD, &request);
        }

        for (int i = 0; i < BLOCK; ++i)             // process the current block; buf[next] must not be touched
            total += buf[cur][i];

        MPI_Wait(&request, MPI_STATUS_IGNORE);      // the next block is complete
        cur = next;
    }

    printf("rank %d: total = %g, expected %g\n", rank, total, BLOCK * NBLOCKS * (NBLOCKS + 1) / 2.0);

    MPI_Finalize();
    return 0;
}
