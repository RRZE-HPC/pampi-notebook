#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <mpi.h>

int main(int argc, char *argv[]) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    double a[100];
    MPI_Aint a1, a2;

    MPI_Get_address(&a[0],  &a1);
    MPI_Get_address(&a[50], &a2);

    printf("displacement of a[50] relative to a[0]: %ld bytes\n", (long)MPI_Aint_diff(a2, a1));

    MPI_Finalize();
    return 0;
}
