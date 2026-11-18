#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <mpi.h>

int main(int argc, char *argv[]) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // every rank sends its rank as an int to the even ranks
    // and its rank + 0.5 as a double to the odd ranks
    struct {
        int    i;
        double d;
    } out = { rank, rank + 0.5 };
    const int disp_i = (char *)&out.i - (char *)&out;   // byte offsets of the members
    const int disp_d = (char *)&out.d - (char *)&out;   // (offsetof() from <stddef.h>)

    int          *sendcounts = malloc(size * sizeof(int));
    int          *sdispls    = malloc(size * sizeof(int));
    MPI_Datatype *sendtypes  = malloc(size * sizeof(MPI_Datatype));
    for (int j = 0; j < size; ++j) {
        sendcounts[j] = 1;
        sendtypes[j]  = j % 2 == 0 ? MPI_INT : MPI_DOUBLE;
        sdispls[j]    = j % 2 == 0 ? disp_i : disp_d;  // in bytes!
    }

    // an even rank receives one int, an odd rank one double from every rank
    MPI_Datatype rtype = rank % 2 == 0 ? MPI_INT : MPI_DOUBLE;
    int rsize;
    MPI_Type_size(rtype, &rsize);
    char         *recvbuf    = malloc(size * rsize);
    int          *recvcounts = malloc(size * sizeof(int));
    int          *rdispls    = malloc(size * sizeof(int));
    MPI_Datatype *recvtypes  = malloc(size * sizeof(MPI_Datatype));
    for (int i = 0; i < size; ++i) {
        recvcounts[i] = 1;
        recvtypes[i]  = rtype;
        rdispls[i]    = i * rsize;                  // in bytes!
    }

    MPI_Alltoallw(&out,    sendcounts, sdispls, sendtypes,
                  recvbuf, recvcounts, rdispls, recvtypes, MPI_COMM_WORLD);

    char line[256];
    int pos = snprintf(line, sizeof(line), "rank %d received", rank);
    for (int i = 0; i < size; ++i) {
        if (rank % 2 == 0)
            pos += snprintf(line + pos, sizeof(line) - pos, " %d",   ((int *)recvbuf)[i]);
        else
            pos += snprintf(line + pos, sizeof(line) - pos, " %.1f", ((double *)recvbuf)[i]);
    }
    printf("%s\n", line);

    free(sendcounts); free(sdispls); free(sendtypes);
    free(recvbuf); free(recvcounts); free(rdispls); free(recvtypes);

    MPI_Finalize();
    return 0;
}
