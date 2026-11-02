#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <mpi.h>

int main(int argc, char *argv[]) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const int N = 1000;

    int start =  rank      * N / size;
    int end   = (rank + 1) * N / size;

    long sum = 0;
    for (int i = start; i < end; ++i)
        sum += i;

    // add up the partial sums of all ranks, the result ends up on rank 0
    long total;
    MPI_Reduce(&sum, &total, 1, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0)
        printf("total = %ld, expected %ld\n", total, (long)N * (N - 1) / 2);

    MPI_Finalize();
    return 0;
}
