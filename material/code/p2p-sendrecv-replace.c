#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <mpi.h>

int main(int argc, char *argv[]) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int left  = (rank - 1 + size) % size;
    int right = (rank + 1) % size;

    int buf[3] = { rank, 10 * rank, 100 * rank };

    MPI_Sendrecv_replace(buf, 3, MPI_INT, right, 0, left, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    printf("rank %d: buf = { %d, %d, %d }\n", rank, buf[0], buf[1], buf[2]);

    MPI_Finalize();
    return 0;
}
