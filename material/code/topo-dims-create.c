#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <mpi.h>

int main(int argc, char *argv[]) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int nprocs[] = { 4, 6, 7, 12, 16, 24 };

    for (int k = 0; k < 6; ++k) {
        int dims2[2] = { 0, 0 }, dims3[3] = { 0, 0, 0 };
        MPI_Dims_create(nprocs[k], 2, dims2);
        MPI_Dims_create(nprocs[k], 3, dims3);
        printf("%3d processes: 2D %2d x %d, 3D %d x %d x %d\n",
               nprocs[k], dims2[0], dims2[1], dims3[0], dims3[1], dims3[2]);
    }

    int fixed[2] = { 0, 2 };                        // 2 processes in dimension 1, MPI chooses dimension 0
    MPI_Dims_create(12, 2, fixed);
    printf(" 12 processes with dims[1] = 2: %d x %d\n", fixed[0], fixed[1]);

    MPI_Finalize();
    return 0;
}
