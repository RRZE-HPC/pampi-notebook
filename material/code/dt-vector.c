#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <mpi.h>

int main(int argc, char *argv[]) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int a[13], b[6];
    for (int i = 0; i < 13; ++i)
        a[i] = i;

    MPI_Datatype nt;
    MPI_Type_vector(2, 3, 5, MPI_INT, &nt);         // 2 blocks of 3 elements, a new block every 5 elements
    MPI_Type_commit(&nt);

    MPI_Sendrecv(a, 1, nt, rank, 0, b, 6, MPI_INT, rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Type_free(&nt);

    printf("received:");
    for (int i = 0; i < 6; ++i)
        printf(" %d", b[i]);
    printf("\n");

    MPI_Finalize();
    return 0;
}
