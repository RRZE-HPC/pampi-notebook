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
        // every rank sends a message of a different length
        int n = 10 * rank;
        double *data = malloc(n * sizeof(double));
        for (int i = 0; i < n; ++i)
            data[i] = rank;
        MPI_Send(data, n, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
        free(data);
    } else {
        for (int i = 1; i < size; ++i) {
            MPI_Status status;
            int n;

            // wait for any message, but don't receive it yet ...
            MPI_Probe(MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);
            MPI_Get_count(&status, MPI_DOUBLE, &n);

            // ... allocate a buffer of the right size and receive exactly the probed message
            double *data = malloc(n * sizeof(double));
            MPI_Recv(data, n, MPI_DOUBLE, status.MPI_SOURCE, status.MPI_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            printf("received %2d doubles from rank %d\n", n, status.MPI_SOURCE);
            free(data);
        }
    }

    MPI_Finalize();
    return 0;
}
