#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <mpi.h>

int main(int argc, char *argv[]) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const int chunk = 2;                            // elements of the result per rank

    int *contrib = malloc(size * chunk * sizeof(int));
    for (int i = 0; i < size * chunk; ++i)
        contrib[i] = rank + i;                      // a contribution to every element

    int part[2];
    MPI_Reduce_scatter_block(contrib, part, chunk, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    // element i of the sum is size * i + (0 + 1 + ... + size-1)
    int i0 = rank * chunk;
    printf("rank %d: elements %d, %d of the sum = { %d, %d }, expected { %d, %d }\n", rank, i0, i0 + 1,
           part[0], part[1], size * i0 + size * (size - 1) / 2, size * (i0 + 1) + size * (size - 1) / 2);

    free(contrib);

    MPI_Finalize();
    return 0;
}
