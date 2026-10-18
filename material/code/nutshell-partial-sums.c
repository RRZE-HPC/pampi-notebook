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

    // determine the chunk [start, end) of this process
    int start =  rank      * N / size;
    int end   = (rank + 1) * N / size;

    long sum = 0;
    for (int i = start; i < end; ++i)
        sum += i;

    printf("rank %d of %d: [%4d, %4d) -> partial sum %ld\n", rank, size, start, end, sum);

    MPI_Finalize();
    return 0;
}
