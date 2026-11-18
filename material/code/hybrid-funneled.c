#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <mpi.h>
#include <omp.h>
#include <math.h>

int main(int argc, char *argv[]) {
    int rank, size, provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    if (provided < MPI_THREAD_FUNNELED) {
        fprintf(stderr, "MPI_THREAD_FUNNELED is not supported\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const int n = 1000000;                          // local vector length
    double *x = malloc(n * sizeof(double));
    for (int i = 0; i < n; ++i)
        x[i] = rank + 1.0;

    double local = 0.0, global = 0.0;               // shared by all threads

    #pragma omp parallel
    {
        for (int iter = 0; iter < 3; ++iter) {
            #pragma omp single
            local = 0.0;                            // implicit barrier at the end of single

            // all threads: the local part of the squared norm
            #pragma omp for reduction(+:local)
            for (int i = 0; i < n; ++i)
                local += x[i] * x[i];               // implicit barrier at the end of the loop

            // only the main thread calls MPI (MPI_THREAD_FUNNELED)
            #pragma omp masked
            {
                MPI_Allreduce(&local, &global, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
                if (rank == 0)
                    printf("iteration %d: norm = %.6f\n", iter, sqrt(global));
            }
            #pragma omp barrier                     // masked has no implied barrier: wait for the result

            // all threads: normalize
            #pragma omp for
            for (int i = 0; i < n; ++i)
                x[i] /= sqrt(global);
        }
    }

    free(x);

    MPI_Finalize();
    return 0;
}
