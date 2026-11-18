#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <mpi.h>
#include <string.h>
#include <math.h>

int main(int argc, char *argv[]) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // arguments: method (lex | rb), omega, n (cells per direction)
    const char *method = argc > 1 ? argv[1] : "rb";
    const double omega = argc > 2 ? atof(argv[2]) : 1.95;
    const int n        = argc > 3 ? atoi(argv[3]) : 64;

    const int imax = n, jmax = n;
    const double dx = 1.0 / imax, dy = 1.0 / jmax, eps = 1e-6;
    const int itermax = 100000;
    const double PI = 3.14159265358979323846;

    // 1D decomposition in j: every rank owns jmaxLocal complete rows, after the jstart rows of the ranks below
    const int jmaxLocal = jmax / size + (rank < jmax % size ? 1 : 0);
    const int jstart    = rank * (jmax / size) + (rank < jmax % size ? rank : jmax % size);
    const int bottom    = rank > 0        ? rank - 1 : MPI_PROC_NULL;
    const int top       = rank < size - 1 ? rank + 1 : MPI_PROC_NULL;

    const size_t bytes = (size_t)(imax + 2) * (jmaxLocal + 2) * sizeof(double);
    double *p = calloc(1, bytes), *rhs = calloc(1, bytes);
    #define P(i, j)   p[(j) * (imax + 2) + (i)]
    #define RHS(i, j) rhs[(j) * (imax + 2) + (i)]

    // local row j is the global row jstart + j
    for (int j = 1; j <= jmaxLocal; ++j)
        for (int i = 1; i <= imax; ++i)
            RHS(i, j) = -2.0 * PI * PI * cos(PI * (i - 0.5) * dx) * cos(PI * (jstart + j - 0.5) * dy);

    // halo exchange of the ghost rows: rows are contiguous, MPI_PROC_NULL handles the global boundary
    #define EXCHANGE()                                                                                  \
        do {                                                                                            \
            MPI_Sendrecv(&P(0, jmaxLocal), imax + 2, MPI_DOUBLE, top, 0,                                \
                         &P(0, 0), imax + 2, MPI_DOUBLE, bottom, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);   \
            MPI_Sendrecv(&P(0, 1), imax + 2, MPI_DOUBLE, bottom, 1,                                     \
                         &P(0, jmaxLocal + 1), imax + 2, MPI_DOUBLE, top, 1, MPI_COMM_WORLD,              \
                         MPI_STATUS_IGNORE);                                                            \
        } while (0)

    const int redblack = strcmp(method, "rb") == 0;
    const double dx2 = dx * dx, dy2 = dy * dy, idx2 = 1.0 / dx2, idy2 = 1.0 / dy2;
    const double factor = omega * 0.5 * dx2 * dy2 / (dx2 + dy2);

    int it = 0;
    double res = 1.0;
    while (res >= eps * eps && it < itermax) {
        res = 0.0;                                  // reset in every iteration!
        if (!redblack)
            EXCHANGE();                             // lexicographic: once per sweep
        for (int pass = 0; pass < (redblack ? 2 : 1); ++pass) {
            if (redblack)
                EXCHANGE();                         // red-black: before each color
            for (int j = 1; j <= jmaxLocal; ++j) {
                // the color depends on the global row index
                int istart = redblack ? 1 + (jstart + j + 1 + pass) % 2 : 1, istep = redblack ? 2 : 1;
                for (int i = istart; i <= imax; i += istep) {
                    double r = RHS(i, j) - ((P(i - 1, j) - 2.0 * P(i, j) + P(i + 1, j)) * idx2 +
                                            (P(i, j - 1) - 2.0 * P(i, j) + P(i, j + 1)) * idy2);
                    P(i, j) -= factor * r;
                    res += r * r;
                }
            }
        }
        // Neumann conditions only at the global boundary
        if (bottom == MPI_PROC_NULL)
            for (int i = 1; i <= imax; ++i) P(i, 0) = P(i, 1);
        if (top == MPI_PROC_NULL)
            for (int i = 1; i <= imax; ++i) P(i, jmaxLocal + 1) = P(i, jmaxLocal);
        for (int j = 1; j <= jmaxLocal; ++j) {
            P(0, j)        = P(1, j);
            P(imax + 1, j) = P(imax, j);
        }
        MPI_Allreduce(MPI_IN_PLACE, &res, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        res /= (double)imax * jmax;
        ++it;
        if (!isfinite(res)) {                       // the iteration diverged
            if (rank == 0)
                printf("%d processes, %-3s omega %.2f n %d: diverged after %d iterations\n", size, method, omega, n, it);
            break;
        }
    }
    if (!isfinite(res)) {
        free(p);
        free(rhs);
        MPI_Finalize();
        return 0;
    }

    // error against the exact solution, after removing the mean
    double mean = 0.0, err = 0.0;
    for (int j = 1; j <= jmaxLocal; ++j)
        for (int i = 1; i <= imax; ++i)
            mean += P(i, j);
    MPI_Allreduce(MPI_IN_PLACE, &mean, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    mean /= (double)imax * jmax;
    for (int j = 1; j <= jmaxLocal; ++j)
        for (int i = 1; i <= imax; ++i) {
            double e = fabs(P(i, j) - mean - cos(PI * (i - 0.5) * dx) * cos(PI * (jstart + j - 0.5) * dy));
            if (e > err) err = e;
        }
    MPI_Allreduce(MPI_IN_PLACE, &err, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    if (rank == 0)
        printf("%d processes, %-3s omega %.2f n %d: %5d iterations, max. error %.3e\n", size, method, omega, n, it, err);
    free(p);
    free(rhs);

    MPI_Finalize();
    return 0;
}
