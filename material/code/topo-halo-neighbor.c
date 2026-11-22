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

    // the neighbor order of a Cartesian topology: for each dimension the - and then the + neighbor,
    // i.e. the order of the results of MPI_Cart_shift() for dimension 0, then for dimension 1
    enum { LEFT, RIGHT, BOTTOM, TOP, NUM_NEIGHBORS };

    // the displacement of G(i, j) in bytes from the start of the grid
    #define DISP(i, j) (((MPI_Aint)(j) * ld + (i)) * (MPI_Aint)sizeof(double))

    // one entry per neighbor, filled by name
    int          counts[NUM_NEIGHBORS]  = { [LEFT]   = 1,      [RIGHT] = 1,
                                            [BOTTOM] = 1,      [TOP]   = 1 };
    MPI_Datatype types[NUM_NEIGHBORS]   = { [LEFT]   = column, [RIGHT] = column,
                                            [BOTTOM] = row,    [TOP]   = row };
    MPI_Aint     sdispls[NUM_NEIGHBORS] = { [LEFT]   = DISP(1, 1),                  // first own column
                                            [RIGHT]  = DISP(imaxLocal, 1),          // last own column
                                            [BOTTOM] = DISP(1, 1),                  // first own row
                                            [TOP]    = DISP(1, jmaxLocal) };        // last own row
    MPI_Aint     rdispls[NUM_NEIGHBORS] = { [LEFT]   = DISP(0, 1),                  // left ghost column
                                            [RIGHT]  = DISP(imaxLocal + 1, 1),      // right ghost column
                                            [BOTTOM] = DISP(1, 0),                  // bottom ghost row
                                            [TOP]    = DISP(1, jmaxLocal + 1) };    // top ghost row

    MPI_Neighbor_alltoallw(grid, counts, sdispls, types, grid, counts, rdispls, types, comm);

    print_grid(comm, grid, imaxLocal, jmaxLocal, 2, 0);

    MPI_Type_free(&column);
    MPI_Type_free(&row);
    MPI_Comm_free(&comm);
    free(grid);

    MPI_Finalize();
    return 0;
}
