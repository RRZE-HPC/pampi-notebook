#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <mpi.h>

int main(int argc, char *argv[]) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank != 0) {
        double t0 = MPI_Wtime();
        while (MPI_Wtime() - t0 < 0.2 * (size - rank))
            ;                                       // "compute" the result
        int result = rank * rank;
        MPI_Send(&result, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
    } else {
        int *results = malloc(size * sizeof(int));
        MPI_Request *requests = malloc(size * sizeof(MPI_Request));

        requests[0] = MPI_REQUEST_NULL;             // nothing to receive from myself
        for (int source = 1; source < size; ++source)
            MPI_Irecv(&results[source], 1, MPI_INT, source, 0, MPI_COMM_WORLD, &requests[source]);

        for (int i = 1; i < size; ++i) {
            int index;
            MPI_Waitany(size, requests, &index, MPI_STATUS_IGNORE);
            printf("rank 0: result %d from rank %d\n", results[index], index);
        }

        free(results);
        free(requests);
    }

    MPI_Finalize();
    return 0;
}
