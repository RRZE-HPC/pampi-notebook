#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <mpi.h>

int main(int argc, char *argv[]) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const int N = 2000;     // dimension of the matrix

    // distribute the rows as evenly as possible: the first N % size ranks get one row more
    const int num = N / size, rest = N % size;
    #define CHUNK_SIZE(r)  (num + ((r) < rest ? 1 : 0))
    #define CHUNK_START(r) ((r) * num + ((r) < rest ? (r) : rest))

    const int nlocal = CHUNK_SIZE(rank);    // number of local rows
    const int start  = CHUNK_START(rank);   // global index of the first local row
    const int nmax   = CHUNK_SIZE(0);       // the largest chunk: x must be able to hold any chunk

    double *a = malloc((size_t)nlocal * N * sizeof(double));    // local rows of A
    double *y = calloc(nlocal, sizeof(double));                // local chunk of y
    double *x = calloc(nmax, sizeof(double));                  // the chunk of x currently held

    // initialize with known values: A_rc = 1 / (r + c + 1), x_c = c + 1
    for (int r = 0; r < nlocal; ++r)
        for (int c = 0; c < N; ++c)
            a[(size_t)r * N + c] = 1.0 / (start + r + c + 1);
    for (int c = 0; c < nlocal; ++c)
        x[c] = start + c + 1;

    int upper = (rank - 1 + size) % size;   // the chunk of x is passed on to this rank ...
    int lower = (rank + 1) % size;          // ... and the next one comes from this rank
    int owner = rank;                       // the rank the current chunk of x belongs to

    // time only the MVM, not the setup and the check
    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();

    for (int step = 0; step < size; ++step) {
        // multiply the matching block of A with the current chunk of x
        const int cs = CHUNK_START(owner), nc = CHUNK_SIZE(owner);
        for (int r = 0; r < nlocal; ++r)
            for (int c = 0; c < nc; ++c)
                y[r] += a[(size_t)r * N + cs + c] * x[c];

        // ring shift of x - not needed after the last step
        if (step < size - 1) {
            MPI_Sendrecv_replace(x, nmax, MPI_DOUBLE, upper, 0, lower, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            owner = (owner + 1) % size;
        }
    }

    double t1 = MPI_Wtime();

    // check the local rows against a serial computation
    double maxerr = 0.0;
    for (int r = 0; r < nlocal; ++r) {
        double ref = 0.0;
        for (int c = 0; c < N; ++c)
            ref += 1.0 / (start + r + c + 1) * (c + 1);
        double err = (y[r] - ref) / ref;
        if (err < 0.0) err = -err;
        if (err > maxerr) maxerr = err;
    }
    printf("rank %d: rows [%4d, %4d), max. relative error %.1e, MVM %.3f ms\n",
           rank, start, start + nlocal, maxerr, 1e3 * (t1 - t0));

    free(a);
    free(y);
    free(x);

    MPI_Finalize();
    return 0;
}
