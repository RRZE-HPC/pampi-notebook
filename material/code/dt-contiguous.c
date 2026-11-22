#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <mpi.h>

int main(int argc, char *argv[]) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int a[6] = { 0, 1, 2, 3, 4, 5 };
    int b[6] = { -1, -1, -1, -1, -1, -1 };

    MPI_Datatype triple;
    MPI_Type_contiguous(3, MPI_INT, &triple);
    MPI_Type_commit(&triple);

    // send 2 triples to myself and receive them as 6 ints: the type signatures match
    MPI_Sendrecv(a, 2, triple, rank, 0, b, 6, MPI_INT, rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    MPI_Type_free(&triple);

    printf("b = { %d, %d, %d, %d, %d, %d }\n", b[0], b[1], b[2], b[3], b[4], b[5]);

    MPI_Finalize();
    return 0;
}
