#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <mpi.h>
#include <string.h>

int main(int argc, char *argv[]) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    char line[100];
    snprintf(line, sizeof(line), "rank %d says hello", rank);

    if (rank != 0) {
        MPI_Send(line, strlen(line) + 1, MPI_CHAR, 0, 0, MPI_COMM_WORLD);   // including the terminating '\0'
    } else {
        printf("%s\n", line);
        for (int source = 1; source < size; ++source) {
            MPI_Recv(line, sizeof(line), MPI_CHAR, source, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            printf("%s\n", line);
        }
    }

    MPI_Finalize();
    return 0;
}
