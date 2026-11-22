#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <mpi.h>

int main(int argc, char *argv[]) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    MPI_Datatype types[3];
    const char *names[3] = { "vector(2, 3, 5, MPI_INT)", "column of a 6 x 5 int matrix", "{(double, 0), (char, 8)}" };

    MPI_Type_vector(2, 3, 5, MPI_INT, &types[0]);
    MPI_Type_vector(6, 1, 5, MPI_INT, &types[1]);

    int          blocklengths[2] = { 1, 1 };
    MPI_Aint     displs[2]       = { 0, 8 };
    MPI_Datatype oldtypes[2]     = { MPI_DOUBLE, MPI_CHAR };
    MPI_Type_create_struct(2, blocklengths, displs, oldtypes, &types[2]);

    for (int k = 0; k < 3; ++k) {
        int nbytes;
        MPI_Aint lb, extent, true_lb, true_extent;
        MPI_Type_size(types[k], &nbytes);
        MPI_Type_get_extent(types[k], &lb, &extent);
        MPI_Type_get_true_extent(types[k], &true_lb, &true_extent);
        printf("%-30s size %3d, lb %ld, extent %3ld, true extent %3ld bytes\n",
               names[k], nbytes, (long)lb, (long)extent, (long)true_extent);
        MPI_Type_free(&types[k]);
    }

    MPI_Finalize();
    return 0;
}
