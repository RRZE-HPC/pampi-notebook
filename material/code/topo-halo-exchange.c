#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <mpi.h>
#include <topo-print-grid.h>

int main(int argc, char *argv[]) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // the global grid: imax x jmax inner cells
    const int imax = 8, jmax = 6;

    // 2D process grid, not periodic; dimension 0 runs along i, dimension 1 along j
    enum { DIM_I, DIM_J, NUM_DIMS };
    int dims[NUM_DIMS] = { 0, 0 }, periods[NUM_DIMS] = { 0, 0 }, coords[NUM_DIMS];
    MPI_Dims_create(size, NUM_DIMS, dims);
    MPI_Comm comm;
    MPI_Cart_create(MPI_COMM_WORLD, NUM_DIMS, dims, periods, 0, &comm);
    MPI_Comm_rank(comm, &rank);
    MPI_Cart_coords(comm, rank, NUM_DIMS, coords);

    // the neighbors: MPI_PROC_NULL at the boundary of the global grid
    int left, right, bottom, top;
    MPI_Cart_shift(comm, DIM_I, 1, &left, &right);
    MPI_Cart_shift(comm, DIM_J, 1, &bottom, &top);

    // the size of the local tile - computed from the coordinates, not from the rank!
    #define SIZE_OF(c, n, p) ((n) / (p) + ((c) < (n) % (p) ? 1 : 0))
    const int imaxLocal = SIZE_OF(coords[DIM_I], imax, dims[DIM_I]);
    const int jmaxLocal = SIZE_OF(coords[DIM_J], jmax, dims[DIM_J]);
    const int ld = imaxLocal + 2;                   // leading dimension: length of a row incl. ghost cells = stride from row j to row j + 1

    // the local grid with one layer of ghost cells, stored row by row
    double *grid = malloc((size_t)ld * (jmaxLocal + 2) * sizeof(double));
    #define G(i, j) grid[(j) * ld + (i)]

    // derived types for the ghost layers
    MPI_Datatype column, row;
    MPI_Type_vector(jmaxLocal, 1, ld, MPI_DOUBLE, &column);     // left and right
    MPI_Type_contiguous(imaxLocal, MPI_DOUBLE, &row);           // bottom and top
    MPI_Type_commit(&column);
    MPI_Type_commit(&row);

    // for debugging: own cells hold the rank, ghost cells -1
    for (int j = 0; j < jmaxLocal + 2; ++j)
        for (int i = 0; i < ld; ++i)
            G(i, j) = (i == 0 || i == ld - 1 || j == 0 || j == jmaxLocal + 1) ? -1 : rank;

    // the tags: the direction of travel of a message
    enum { UPWARD, DOWNWARD, RIGHTWARD, LEFTWARD };

    // the requests: one receive from and one send to every neighbor
    enum { RECV_BOTTOM, RECV_TOP, RECV_LEFT, RECV_RIGHT,
           SEND_TOP, SEND_BOTTOM, SEND_RIGHT, SEND_LEFT, NUM_REQUESTS };

    // exchange: receive into the ghost layers, send the adjacent boundary layers
    MPI_Request requests[NUM_REQUESTS];
    MPI_Irecv(&G(1, 0),             1, row,    bottom, UPWARD,    comm, &requests[RECV_BOTTOM]);
    MPI_Irecv(&G(1, jmaxLocal + 1), 1, row,    top,    DOWNWARD,  comm, &requests[RECV_TOP]);
    MPI_Irecv(&G(0, 1),             1, column, left,   RIGHTWARD, comm, &requests[RECV_LEFT]);
    MPI_Irecv(&G(imaxLocal + 1, 1), 1, column, right,  LEFTWARD,  comm, &requests[RECV_RIGHT]);
    MPI_Isend(&G(1, jmaxLocal),     1, row,    top,    UPWARD,    comm, &requests[SEND_TOP]);
    MPI_Isend(&G(1, 1),             1, row,    bottom, DOWNWARD,  comm, &requests[SEND_BOTTOM]);
    MPI_Isend(&G(imaxLocal, 1),     1, column, right,  RIGHTWARD, comm, &requests[SEND_RIGHT]);
    MPI_Isend(&G(1, 1),             1, column, left,   LEFTWARD,  comm, &requests[SEND_LEFT]);
    MPI_Waitall(NUM_REQUESTS, requests, MPI_STATUSES_IGNORE);

    print_grid(comm, grid, imaxLocal, jmaxLocal, 2, 0);

    MPI_Type_free(&column);
    MPI_Type_free(&row);
    MPI_Comm_free(&comm);
    free(grid);

    MPI_Finalize();
    return 0;
}
