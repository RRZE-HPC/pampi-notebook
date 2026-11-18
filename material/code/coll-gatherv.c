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

    int n = rank + 1;                               // rank r contributes r + 1 elements
    int *mine = malloc(n * sizeof(int));
    for (int i = 0; i < n; ++i)
        mine[i] = rank;

    int *recvcounts = NULL, *displs = NULL, *all = NULL;
    if (rank == root) {
        recvcounts = malloc(size * sizeof(int));
        displs     = malloc(size * sizeof(int));
        int total = 0;
        for (int r = 0; r < size; ++r) {
            recvcounts[r] = r + 1;
            displs[r]     = total;                  // the chunks are stored back to back
            total        += recvcounts[r];
        }
        all = malloc(total * sizeof(int));
    }

    MPI_Gatherv(mine, n, MPI_INT, all, recvcounts, displs, MPI_INT, root, MPI_COMM_WORLD);

    if (rank == root) {
        printf("root: all =");
        for (int i = 0; i < displs[size - 1] + recvcounts[size - 1]; ++i)
            printf(" %d", all[i]);
        printf("\n");
        free(recvcounts);
        free(displs);
        free(all);
    }
    free(mine);

    MPI_Finalize();
    return 0;
}
