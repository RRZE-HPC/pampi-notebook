#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <mpi.h>

int main(int argc, char *argv[]) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const int msgs = 1000;      // messages per sender

    if (rank != 0) {
        for (int i = 0; i < msgs; ++i)
            MPI_Send(&rank, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
    } else {
        int *count = calloc(size, sizeof(int));
        int source;

        // the first msgs messages: whoever comes first ...
        for (int i = 0; i < msgs; ++i) {
            MPI_Recv(&source, 1, MPI_INT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            ++count[source];
        }
        for (int r = 1; r < size; ++r)
            printf("%4d of the first %d messages came from rank %d\n", count[r], msgs, r);

        // ... and all remaining ones, so that every sender can finish
        for (int i = msgs; i < (size - 1) * msgs; ++i)
            MPI_Recv(&source, 1, MPI_INT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        free(count);
    }

    MPI_Finalize();
    return 0;
}
