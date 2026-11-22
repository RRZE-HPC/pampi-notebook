#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <mpi.h>

int main(int argc, char *argv[]) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const int nrows = 6, ncols = 5;

    int m[6 * 5], b[12];
    for (int i = 0; i < nrows * ncols; ++i)
        m[i] = i;

    int sizes[2]    = { nrows, ncols };             // C order: the slowest dimension first
    int subsizes[2] = { nrows - 2, ncols - 2 };
    int starts[2]   = { 1, 1 };

    MPI_Datatype bulk;
    MPI_Type_create_subarray(2, sizes, subsizes, starts, MPI_ORDER_C, MPI_INT, &bulk);
    MPI_Type_commit(&bulk);

    MPI_Aint lb, extent;
    MPI_Type_get_extent(bulk, &lb, &extent);
    printf("extent of the subarray type: %ld bytes\n", (long)extent);

    // the buffer is the start of the full array
    MPI_Sendrecv(m, 1, bulk, rank, 0, b, 12, MPI_INT, rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Type_free(&bulk);

    for (int r = 0; r < nrows - 2; ++r) {
        for (int c = 0; c < ncols - 2; ++c)
            printf(" %2d", b[r * (ncols - 2) + c]);
        printf("\n");
    }

    MPI_Finalize();
    return 0;
}
