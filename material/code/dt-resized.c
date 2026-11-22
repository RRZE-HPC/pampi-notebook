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

    // the matrix, followed by as many elements of -1 - room for reading beyond the matrix
    int *m = malloc(2 * nrows * ncols * sizeof(int));
    for (int i = 0; i < 2 * nrows * ncols; ++i)
        m[i] = i < nrows * ncols ? i : -1;
    int b[12];

    MPI_Datatype column, column1;
    MPI_Type_vector(nrows, 1, ncols, MPI_INT, &column);
    MPI_Type_commit(&column);

    // count = 2: the 2nd copy starts one extent (26 ints) after the 1st
    MPI_Sendrecv(&m[1], 2, column, rank, 0, b, 12, MPI_INT, rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    printf("column, count = 2:            ");
    for (int i = 0; i < 12; ++i)
        printf(" %2d", b[i]);
    printf("\n");

    // the same type with an extent of one int: the 2nd copy is the next column
    MPI_Type_create_resized(column, 0, sizeof(int), &column1);
    MPI_Type_commit(&column1);
    MPI_Sendrecv(&m[1], 2, column1, rank, 0, b, 12, MPI_INT, rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    printf("resized column, count = 2:    ");
    for (int i = 0; i < 12; ++i)
        printf(" %2d", b[i]);
    printf("\n");

    MPI_Type_free(&column);
    MPI_Type_free(&column1);
    free(m);

    MPI_Finalize();
    return 0;
}
