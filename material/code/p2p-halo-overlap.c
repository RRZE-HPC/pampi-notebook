#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <mpi.h>

int main(int argc, char *argv[]) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    #define N 6     // own cells per rank

    double u[N + 2], unew[N + 2];               // u[0] and u[N + 1] are ghost cells
    for (int i = 1; i <= N; ++i)
        u[i] = rank * N + i - 1;                // a linear function: the global index

    int left  = rank > 0        ? rank - 1 : MPI_PROC_NULL;
    int right = rank < size - 1 ? rank + 1 : MPI_PROC_NULL;

    // bogus ghost values - except at the global boundary (linear extrapolation)
    u[0]     = left  == MPI_PROC_NULL ? -1.0     : -1000.0;
    u[N + 1] = right == MPI_PROC_NULL ? size * N : -1000.0;

    // 1. copy to send buffers: not needed, the boundary values are single numbers
    // 2. start all communication - tag 0: data moving to the right, tag 1: data moving to the left
    MPI_Request requests[4];
    MPI_Irecv(&u[0],     1, MPI_DOUBLE, left,  0, MPI_COMM_WORLD, &requests[0]);
    MPI_Irecv(&u[N + 1], 1, MPI_DOUBLE, right, 1, MPI_COMM_WORLD, &requests[1]);
    MPI_Isend(&u[1],     1, MPI_DOUBLE, left,  1, MPI_COMM_WORLD, &requests[2]);
    MPI_Isend(&u[N],     1, MPI_DOUBLE, right, 0, MPI_COMM_WORLD, &requests[3]);

    // 3. bulk update: only reads u[1 .. N] - reading a send buffer is allowed
    for (int i = 2; i <= N - 1; ++i)
        unew[i] = (u[i - 1] + u[i] + u[i + 1]) / 3.0;

    // 4. wait for all communication to complete
    MPI_Waitall(4, requests, MPI_STATUSES_IGNORE);

    // 5. copy into ghost cells: not needed, they were received in place
    // 6. update the cells next to the ghost cells
    unew[1] = (u[0]     + u[1] + u[2])     / 3.0;
    unew[N] = (u[N - 1] + u[N] + u[N + 1]) / 3.0;

    double maxdev = 0.0;
    for (int i = 1; i <= N; ++i) {
        double dev = unew[i] - u[i];
        if (dev < 0.0) dev = -dev;
        if (dev > maxdev) maxdev = dev;
    }
    printf("rank %d: ghost cells %5.1f %5.1f, max. deviation %g\n", rank, u[0], u[N + 1], maxdev);

    MPI_Finalize();
    return 0;
}
