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

    int data[4][4] = { { 0, 9, 2, 6 }, { 5, 1, 0, 4 }, { 8, 3, 4, 5 }, { 1, 0, 6, 8 } };
    int max[4], sum[4];

    // element-wise maximum, only on the root
    MPI_Reduce(data[rank], max, 4, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0)
        printf("rank 0: max = { %d, %d, %d, %d }\n", max[0], max[1], max[2], max[3]);

    // element-wise sum, on every rank
    MPI_Allreduce(data[rank], sum, 4, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    printf("rank %d: sum = { %d, %d, %d, %d }\n", rank, sum[0], sum[1], sum[2], sum[3]);

    MPI_Finalize();
    return 0;
}
