#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <mpi.h>

int main(int argc, char *argv[]) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    #define N 4     // own cells per rank

    double u[N + 2];                            // u[0] and u[N + 1] are ghost cells
    u[0] = u[N + 1] = -1.0;                     // marks ghost cells that were not updated
    for (int i = 1; i <= N; ++i)
        u[i] = rank * N + i - 1;                // the global index of the cell

    int left  = rank > 0        ? rank - 1 : MPI_PROC_NULL;
    int right = rank < size - 1 ? rank + 1 : MPI_PROC_NULL;

    // send my last own cell to the right, receive my left ghost cell from the left ...
    MPI_Sendrecv(&u[N], 1, MPI_DOUBLE, right, 0,
                 &u[0], 1, MPI_DOUBLE, left,  0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    // ... and the other way round
    MPI_Sendrecv(&u[1],     1, MPI_DOUBLE, left,  1,
                 &u[N + 1], 1, MPI_DOUBLE, right, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    // prints exactly N = 4 own cells
    printf("rank %d: %5.1f | %4.1f %4.1f %4.1f %4.1f | %5.1f\n",
           rank, u[0], u[1], u[2], u[3], u[4], u[5]);

    MPI_Finalize();
    return 0;
}
