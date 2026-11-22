#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <mpi.h>

int main(int argc, char *argv[]) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    double a[14], b[8];
    for (int i = 0; i < 14; ++i)
        a[i] = i;

    MPI_Datatype pair, nt;
    MPI_Type_contiguous(2, MPI_DOUBLE, &pair);                      // {(double, 0), (double, 8)}, extent 16

    int blocklengths[2]  = { 3, 1 };
    int displacements[2] = { 4, 0 };                                // in units of the extent of pair
    MPI_Type_indexed(2, blocklengths, displacements, pair, &nt);
    MPI_Type_commit(&nt);

    int nbytes;
    MPI_Aint lb, extent;
    MPI_Type_size(nt, &nbytes);
    MPI_Type_get_extent(nt, &lb, &extent);
    printf("size %d bytes, lb %ld, extent %ld bytes\n", nbytes, (long)lb, (long)extent);

    MPI_Sendrecv(a, 1, nt, rank, 0, b, 8, MPI_DOUBLE, rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    printf("selected:");
    for (int i = 0; i < 8; ++i)
        printf(" %g", b[i]);
    printf("\n");

    MPI_Type_free(&pair);
    MPI_Type_free(&nt);

    MPI_Finalize();
    return 0;
}
