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

    MPI_Datatype column;
    MPI_Type_vector(nrows, 1, ncols, MPI_INT, &column);
    MPI_Type_commit(&column);

    if (rank == 0) {
        int m[6][5];
        for (int r = 0; r < nrows; ++r)
            for (int c = 0; c < ncols; ++c)
                m[r][c] = r * ncols + c;

        MPI_Send(&m[0][1], 1, column, 1, 0, MPI_COMM_WORLD);     // column 1
    } else if (rank == 1) {
        int col[6];
        MPI_Recv(col, nrows, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        printf("rank 1: column =");
        for (int r = 0; r < nrows; ++r)
            printf(" %d", col[r]);
        printf("\n");
    }

    MPI_Type_free(&column);

    MPI_Finalize();
    return 0;
}
