#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <mpi.h>

int main(int argc, char *argv[]) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int position = 0;

    if (rank == 0) {
        int n = 5;
        double values[5] = { 1.5, 2.5, 3.5, 4.5, 5.5 };

        // an upper bound for the packed size of the count and the values
        int size_n, size_values;
        MPI_Pack_size(1, MPI_INT,    MPI_COMM_WORLD, &size_n);
        MPI_Pack_size(n, MPI_DOUBLE, MPI_COMM_WORLD, &size_values);
        const int bufsize = size_n + size_values;
        char *buffer = malloc(bufsize);

        MPI_Pack(&n, 1, MPI_INT, buffer, bufsize, &position, MPI_COMM_WORLD);
        MPI_Pack(values, n, MPI_DOUBLE, buffer, bufsize, &position, MPI_COMM_WORLD);
        MPI_Send(buffer, position, MPI_PACKED, 1, 0, MPI_COMM_WORLD);
        printf("rank 0: sent %d packed bytes\n", position);
        free(buffer);
    } else if (rank == 1) {
        // the receiver does not know the size of the message: probe it first
        MPI_Status status;
        int nbytes;
        MPI_Probe(0, 0, MPI_COMM_WORLD, &status);
        MPI_Get_count(&status, MPI_PACKED, &nbytes);
        char *buffer = malloc(nbytes);
        MPI_Recv(buffer, nbytes, MPI_PACKED, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        // first the count, then as many values
        int n;
        MPI_Unpack(buffer, nbytes, &position, &n, 1, MPI_INT, MPI_COMM_WORLD);
        double *values = malloc(n * sizeof(double));
        MPI_Unpack(buffer, nbytes, &position, values, n, MPI_DOUBLE, MPI_COMM_WORLD);

        printf("rank 1: %d values:", n);
        for (int i = 0; i < n; ++i)
            printf(" %g", values[i]);
        printf("\n");
        free(values);
        free(buffer);
    }

    MPI_Finalize();
    return 0;
}
