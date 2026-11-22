#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <mpi.h>

int main(int argc, char *argv[]) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int    i = 0x01020304;
    double d = 1.0;
    unsigned char native[64], external[64];
    int position = 0;
    MPI_Aint xposition = 0;

    // the same data, packed in the native and in the portable external32 representation
    MPI_Pack(&i, 1, MPI_INT,    native, sizeof native, &position, MPI_COMM_WORLD);
    MPI_Pack(&d, 1, MPI_DOUBLE, native, sizeof native, &position, MPI_COMM_WORLD);
    MPI_Pack_external("external32", &i, 1, MPI_INT,    external, sizeof external, &xposition);
    MPI_Pack_external("external32", &d, 1, MPI_DOUBLE, external, sizeof external, &xposition);

    printf("native     (%2d bytes):", position);
    for (int k = 0; k < position; ++k)
        printf(" %02x", native[k]);
    printf("\nexternal32 (%2ld bytes):", (long)xposition);
    for (int k = 0; k < xposition; ++k)
        printf(" %02x", external[k]);
    printf("\n");

    MPI_Finalize();
    return 0;
}
