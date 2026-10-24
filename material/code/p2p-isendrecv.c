#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <mpi.h>

int main(int argc, char *argv[]) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    #if MPI_VERSION >= 4 || (defined(OMPI_MAJOR_VERSION) && OMPI_MAJOR_VERSION >= 5)
    int left  = (rank - 1 + size) % size;
    int right = (rank + 1) % size;
    int sendval = rank, recvval = -1;
    MPI_Request request;

    MPI_Isendrecv(&sendval, 1, MPI_INT, right, 0,
                  &recvval, 1, MPI_INT, left,  0, MPI_COMM_WORLD, &request);
    // work that does not touch sendval or recvval could be done here
    MPI_Wait(&request, MPI_STATUS_IGNORE);

    printf("rank %d: received %d\n", rank, recvval);
    #else
    if (rank == 0)
        printf("MPI_Isendrecv() requires MPI 4.0, this library implements MPI %d.%d\n", MPI_VERSION, MPI_SUBVERSION);
    #endif

    MPI_Finalize();
    return 0;
}
