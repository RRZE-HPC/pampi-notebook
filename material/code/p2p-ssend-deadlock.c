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
    int sendval = rank, recvval;

    MPI_Ssend(&sendval, 1, MPI_INT, other, 0, MPI_COMM_WORLD);
    MPI_Recv(&recvval, 1, MPI_INT, other, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    printf("rank %d: received %d\n", rank, recvval);

    MPI_Finalize();
    return 0;
}
