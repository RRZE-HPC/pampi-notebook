#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <mpi.h>

int main(int argc, char *argv[]) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    #define MAX_COUNT 100

    if (rank != 0) {
        int buf[MAX_COUNT];
        for (int i = 0; i < rank; ++i)
            buf[i] = rank;

        MPI_Send(buf, rank, MPI_INT, 0, 0, MPI_COMM_WORLD);
    } else {
        for (int i = 1; i < size; ++i) {
            int buf[MAX_COUNT], count;
            MPI_Status s;

            MPI_Recv(buf, MAX_COUNT, MPI_INT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &s);
            MPI_Get_count(&s, MPI_INT, &count);

            printf("Received %d elements from rank %d\n", count, s.MPI_SOURCE);
        }
    }

    MPI_Finalize();
    return 0;
}
