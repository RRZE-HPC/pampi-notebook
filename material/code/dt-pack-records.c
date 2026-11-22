#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <mpi.h>
#include <string.h>

int main(int argc, char *argv[]) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // a record: a name, then a number of values; the records differ in both
    struct record { const char *name; int n; double values[4]; };

    if (rank == 0) {
        struct record rec[3] = {
            { "pressure",    1, { 101.3 } },
            { "velocity",    3, { 1.0, -0.5, 0.25 } },
            { "temperature", 2, { 293.15, 295.0 } },
        };

        // an upper bound for the packed size, record by record
        int bufsize = 0;
        for (int r = 0; r < 3; ++r) {
            int s;
            int len = strlen(rec[r].name);
            MPI_Pack_size(1, MPI_INT, MPI_COMM_WORLD, &s);              // len and n: two MPI_Pack() calls
            bufsize += 2 * s;
            MPI_Pack_size(len, MPI_CHAR, MPI_COMM_WORLD, &s);           // the name
            bufsize += s;
            MPI_Pack_size(rec[r].n, MPI_DOUBLE, MPI_COMM_WORLD, &s);    // the values
            bufsize += s;
        }
        char *buffer = malloc(bufsize);

        int position = 0;
        for (int r = 0; r < 3; ++r) {
            int len = strlen(rec[r].name);
            MPI_Pack(&len,          1,        MPI_INT,    buffer, bufsize, &position, MPI_COMM_WORLD);
            MPI_Pack(rec[r].name,   len,      MPI_CHAR,   buffer, bufsize, &position, MPI_COMM_WORLD);
            MPI_Pack(&rec[r].n,     1,        MPI_INT,    buffer, bufsize, &position, MPI_COMM_WORLD);
            MPI_Pack(rec[r].values, rec[r].n, MPI_DOUBLE, buffer, bufsize, &position, MPI_COMM_WORLD);
        }
        printf("rank 0: packed 3 records into %d of %d bytes\n", position, bufsize);
        MPI_Send(buffer, position, MPI_PACKED, 1, 0, MPI_COMM_WORLD);
        free(buffer);
    } else if (rank == 1) {
        MPI_Status status;
        int nbytes;
        MPI_Probe(0, 0, MPI_COMM_WORLD, &status);
        MPI_Get_count(&status, MPI_PACKED, &nbytes);
        char *buffer = malloc(nbytes);
        MPI_Recv(buffer, nbytes, MPI_PACKED, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        // the receiver neither knows how many records there are nor how long they are:
        // it unpacks until the cursor reaches the end of the message
        int position = 0;
        while (position < nbytes) {
            // every length is unpacked before the data it describes: allocate just in time
            int len, n;
            MPI_Unpack(buffer, nbytes, &position, &len,  1,   MPI_INT,    MPI_COMM_WORLD);
            char *name = malloc(len + 1);
            MPI_Unpack(buffer, nbytes, &position, name,  len, MPI_CHAR,   MPI_COMM_WORLD);
            name[len] = '\0';
            MPI_Unpack(buffer, nbytes, &position, &n,    1,   MPI_INT,    MPI_COMM_WORLD);
            double *values = malloc(n * sizeof(double));
            MPI_Unpack(buffer, nbytes, &position, values, n,  MPI_DOUBLE, MPI_COMM_WORLD);

            printf("rank 1: %-12s", name);
            for (int i = 0; i < n; ++i)
                printf(" %8g", values[i]);
            printf("   (position %d)\n", position);
            free(name);
            free(values);
        }
        free(buffer);
    }

    MPI_Finalize();
    return 0;
}
