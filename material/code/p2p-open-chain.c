#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <mpi.h>

int main(int argc, char *argv[]) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int left  = rank - 1;    if (left < 0)      left  = MPI_PROC_NULL;
    int right = rank + 1;    if (right >= size) right = MPI_PROC_NULL;

    int sendval = 10 * rank, recvval = -1;
    MPI_Status status;

    MPI_Sendrecv(&sendval, 1, MPI_INT, right, 0,
                 &recvval, 1, MPI_INT, left,  0, MPI_COMM_WORLD, &status);

    int count;
    MPI_Get_count(&status, MPI_INT, &count);
    printf("rank %d: recvval = %2d, count = %d%s\n", rank, recvval, count,
           status.MPI_SOURCE == MPI_PROC_NULL ? " (received from MPI_PROC_NULL)" : "");

    MPI_Finalize();
    return 0;
}
