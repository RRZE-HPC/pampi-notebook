// Debugging helper for 2D domain decompositions: collects the local grids of all ranks of a
// 2D Cartesian communicator on rank 0 and prints them side by side, ghost cells included,
// with the top row first. The local grid has (imaxLocal + 2) x (jmaxLocal + 2) cells, stored
// row by row: grid[j * (imaxLocal + 2) + i]. Dimension 0 of the communicator runs along i,
// dimension 1 along j. Uses the tag 9999.
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

static void print_grid(MPI_Comm comm, const double *grid, int imaxLocal, int jmaxLocal, int width, int precision) {
    // the dimensions of the process grid: dimension 0 runs along i, dimension 1 along j
    enum { DIM_I, DIM_J, NUM_DIMS };
    int rank, size, dims[NUM_DIMS], periods[NUM_DIMS], coords[NUM_DIMS];
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
    MPI_Cart_get(comm, NUM_DIMS, dims, periods, coords);

    // rank 0 needs the size of every tile
    int local[NUM_DIMS] = { [DIM_I] = imaxLocal + 2, [DIM_J] = jmaxLocal + 2 }, *tiles = NULL;
    if (rank == 0)
        tiles = malloc(NUM_DIMS * size * sizeof(int));
    MPI_Gather(local, NUM_DIMS, MPI_INT, tiles, NUM_DIMS, MPI_INT, 0, comm);

    MPI_Request request;
    MPI_Isend(grid, local[DIM_I] * local[DIM_J], MPI_DOUBLE, 0, 9999, comm, &request);

    if (rank == 0) {
        // the width of every column and the height of every row of tiles
        int *wi = calloc(dims[DIM_I], sizeof(int)), *hj = calloc(dims[DIM_J], sizeof(int));
        for (int r = 0; r < size; ++r) {
            int c[NUM_DIMS];
            MPI_Cart_coords(comm, r, NUM_DIMS, c);
            wi[c[DIM_I]] = tiles[NUM_DIMS * r + DIM_I];
            hj[c[DIM_J]] = tiles[NUM_DIMS * r + DIM_J];
        }

        // the canvas holds all tiles with a gap of one cell in between; NAN marks the gaps
        int W = dims[DIM_I] - 1, H = dims[DIM_J] - 1;
        for (int k = 0; k < dims[DIM_I]; ++k) W += wi[k];
        for (int k = 0; k < dims[DIM_J]; ++k) H += hj[k];
        double *canvas = malloc((size_t)W * H * sizeof(double));
        for (int k = 0; k < W * H; ++k)
            canvas[k] = NAN;

        // receive every tile directly into its place in the canvas, with a subarray type
        for (int r = 0; r < size; ++r) {
            int c[NUM_DIMS], oi = 0, oj = 0;
            MPI_Cart_coords(comm, r, NUM_DIMS, c);
            for (int k = 0; k < c[DIM_I]; ++k) oi += wi[k] + 1;
            for (int k = 0; k < c[DIM_J]; ++k) oj += hj[k] + 1;

            int sizes[2] = { H, W }, subsizes[2] = { hj[c[DIM_J]], wi[c[DIM_I]] }, starts[2] = { oj, oi };
            MPI_Datatype tile;
            MPI_Type_create_subarray(2, sizes, subsizes, starts, MPI_ORDER_C, MPI_DOUBLE, &tile);
            MPI_Type_commit(&tile);
            MPI_Recv(canvas, 1, tile, r, 9999, comm, MPI_STATUS_IGNORE);
            MPI_Type_free(&tile);
        }

        for (int j = H - 1; j >= 0; --j) {
            for (int i = 0; i < W; ++i) {
                double v = canvas[(size_t)j * W + i];
                if (isnan(v))
                    printf("%*s", width + 1, "");
                else
                    printf(" %*.*f", width, precision, v);
            }
            printf("\n");
        }
        free(canvas);
        free(wi);
        free(hj);
        free(tiles);
    }
    MPI_Wait(&request, MPI_STATUS_IGNORE);
}
