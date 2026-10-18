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
        int value = rank * rank;
        MPI_Send(&value, 1, MPI_INT, 0, rank, MPI_COMM_WORLD);
    } else {
        for (int i = 1; i < size; ++i) {
            int value;
            MPI_Status s;
            MPI_Recv(&value, 1, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &s);
            printf("Received %2d from rank %d with tag %d\n", value, s.MPI_SOURCE, s.MPI_TAG);
        }
    }

    MPI_Finalize();
    return 0;
}
