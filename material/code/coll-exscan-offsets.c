#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <mpi.h>

int main(int argc, char *argv[]) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int n = rank % 3 + 1;                           // a different number of elements on every rank

    int offset, total;
    MPI_Exscan(&n, &offset, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    if (rank == 0)
        offset = 0;
    MPI_Allreduce(&n, &total, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    printf("rank %d: %d elements, global indices [%d, %d) of %d\n", rank, n, offset, offset + n, total);

    MPI_Finalize();
    return 0;
}
