#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <mpi.h>

int main(int argc, char *argv[]) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const int root = 0;
    const int chunk = 2;            // elements per rank

    int *data = NULL;               // the complete array, only needed on the root
    if (rank == root) {
        data = malloc(size * chunk * sizeof(int));
        for (int i = 0; i < size * chunk; ++i)
            data[i] = i;
    }

    int local[2];
    MPI_Scatter(data, chunk, MPI_INT, local, chunk, MPI_INT, root, MPI_COMM_WORLD);
    printf("rank %d: got { %d, %d }\n", rank, local[0], local[1]);

    for (int i = 0; i < chunk; ++i)
        local[i] *= 10;             // work on the local chunk

    MPI_Gather(local, chunk, MPI_INT, data, chunk, MPI_INT, root, MPI_COMM_WORLD);

    if (rank == root) {
        printf("root: data =");
        for (int i = 0; i < size * chunk; ++i)
            printf(" %d", data[i]);
        printf("\n");
        free(data);
    }

    MPI_Finalize();
    return 0;
}
