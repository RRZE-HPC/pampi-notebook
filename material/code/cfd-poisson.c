#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <string.h>
#include <math.h>

int main(int argc, char *argv[]) {
    // arguments: method (jacobi | gs | sor | rb), omega, n (cells per direction)
    const char *method = argc > 1 ? argv[1] : "sor";
    const double omega = argc > 2 ? atof(argv[2]) : 1.9;
    const int n        = argc > 3 ? atoi(argv[3]) : 64;

    const int imax = n, jmax = n;                   // unit square, n x n cells
    const double dx = 1.0 / imax, dy = 1.0 / jmax;
    const double eps = 1e-6;                        // stop when the RMS residual is below eps
    const int itermax = 100000;
    const double PI = 3.14159265358979323846;

    const size_t bytes = (size_t)(imax + 2) * (jmax + 2) * sizeof(double);
    double *p = calloc(1, bytes), *pold = calloc(1, bytes), *rhs = calloc(1, bytes);
    #define P(i, j)    p[(j) * (imax + 2) + (i)]
    #define POLD(i, j) pold[(j) * (imax + 2) + (i)]
    #define RHS(i, j)  rhs[(j) * (imax + 2) + (i)]

    // manufactured solution p = cos(pi x) cos(pi y): homogeneous Neumann conditions, f = -2 pi^2 p
    for (int j = 1; j <= jmax; ++j)
        for (int i = 1; i <= imax; ++i)
            RHS(i, j) = -2.0 * PI * PI * cos(PI * (i - 0.5) * dx) * cos(PI * (j - 0.5) * dy);

    const int jacobi = strcmp(method, "jacobi") == 0, redblack = strcmp(method, "rb") == 0;
    const double w = (jacobi || strcmp(method, "gs") == 0) ? 1.0 : omega;
    const double dx2 = dx * dx, dy2 = dy * dy, idx2 = 1.0 / dx2, idy2 = 1.0 / dy2;
    const double factor = w * 0.5 * dx2 * dy2 / (dx2 + dy2);

    int it = 0;
    double res = 1.0;
    while (res >= eps * eps && it < itermax) {
        res = 0.0;
        if (jacobi)
            memcpy(pold, p, bytes);                 // Jacobi: compute from the old values only
        double *q = jacobi ? pold : p;              // Gauss-Seidel/SOR: always the newest values
    #define Q(i, j) q[(j) * (imax + 2) + (i)]
        for (int pass = 0; pass < (redblack ? 2 : 1); ++pass) {
            for (int j = 1; j <= jmax; ++j) {
                // red-black: in pass 0 the points with i + j even, in pass 1 those with i + j odd
                int istart = redblack ? 1 + (j + 1 + pass) % 2 : 1, istep = redblack ? 2 : 1;
                for (int i = istart; i <= imax; i += istep) {
                    double r = RHS(i, j) - ((Q(i - 1, j) - 2.0 * Q(i, j) + Q(i + 1, j)) * idx2 +
                                            (Q(i, j - 1) - 2.0 * Q(i, j) + Q(i, j + 1)) * idy2);
                    P(i, j) = Q(i, j) - factor * r;
                    res += r * r;
                }
            }
        }
        // homogeneous Neumann boundary conditions in the ghost cells
        for (int i = 1; i <= imax; ++i) {
            P(i, 0)        = P(i, 1);
            P(i, jmax + 1) = P(i, jmax);
        }
        for (int j = 1; j <= jmax; ++j) {
            P(0, j)        = P(1, j);
            P(imax + 1, j) = P(imax, j);
        }
        res /= (double)imax * jmax;                 // mean square residual
        ++it;
    }

    // the solution is unique only up to a constant: compare after removing the mean
    double mean = 0.0, err = 0.0;
    for (int j = 1; j <= jmax; ++j)
        for (int i = 1; i <= imax; ++i)
            mean += P(i, j);
    mean /= (double)imax * jmax;
    for (int j = 1; j <= jmax; ++j)
        for (int i = 1; i <= imax; ++i) {
            double e = fabs(P(i, j) - mean - cos(PI * (i - 0.5) * dx) * cos(PI * (j - 0.5) * dy));
            if (e > err) err = e;
        }
    printf("%-6s omega %.3f n %4d: %6d iterations, max. error %.3e\n", method, w, n, it, err);
    free(p); free(pold); free(rhs);
    return 0;
}
