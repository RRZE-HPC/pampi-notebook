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
    char buffer[256];                       // the pack buffer: large enough for this example
    int position = 0;                       // the cursor: the next free byte in the buffer

    if (rank == 0) {
        int m[6 * 5];
        for (int i = 0; i < nrows * ncols; ++i)
            m[i] = i;

        MPI_Datatype column;
        MPI_Type_vector(nrows, 1, ncols, MPI_INT, &column);
        MPI_Type_commit(&column);

        int j = 2;                          // which column to send
        printf("rank 0: position %3d at the start\n", position);
        MPI_Pack(&j, 1, MPI_INT, buffer, sizeof buffer, &position, MPI_COMM_WORLD);
        printf("rank 0: position %3d after packing the column index\n", position);
        MPI_Pack(&m[j], 1, column, buffer, sizeof buffer, &position, MPI_COMM_WORLD);
        printf("rank 0: position %3d after packing the column\n", position);

        // send exactly the packed bytes
        MPI_Send(buffer, position, MPI_PACKED, 1, 0, MPI_COMM_WORLD);
        MPI_Type_free(&column);
    } else if (rank == 1) {
        MPI_Status status;
        int nbytes;
        MPI_Recv(buffer, sizeof buffer, MPI_PACKED, 0, 0, MPI_COMM_WORLD, &status);
        MPI_Get_count(&status, MPI_PACKED, &nbytes);

        // unpack in the same order: the column index, then the column - into a contiguous array
        int j, col[6];
        MPI_Unpack(buffer, nbytes, &position, &j, 1, MPI_INT, MPI_COMM_WORLD);
        MPI_Unpack(buffer, nbytes, &position, col, nrows, MPI_INT, MPI_COMM_WORLD);

        printf("rank 1: received %d bytes, column %d:", nbytes, j);
        for (int i = 0; i < nrows; ++i)
            printf(" %2d", col[i]);
        printf("\n");
    }

    MPI_Finalize();
    return 0;
}
