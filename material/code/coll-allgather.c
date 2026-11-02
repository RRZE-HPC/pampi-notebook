#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <mpi.h>

int main(int argc, char *argv[]) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int mine = rank * rank;
    int *all = malloc(size * sizeof(int));

    MPI_Allgather(&mine, 1, MPI_INT, all, 1, MPI_INT, MPI_COMM_WORLD);

    char line[256];
    int pos = snprintf(line, sizeof(line), "rank %d: all =", rank);
    for (int i = 0; i < size; ++i)
        pos += snprintf(line + pos, sizeof(line) - pos, " %d", all[i]);
    printf("%s\n", line);

    free(all);

    MPI_Finalize();
    return 0;
}
