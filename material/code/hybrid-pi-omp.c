#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <omp.h>

int main(int argc, char *argv[]) {
    struct timespec ice_start, ice_end;
    clock_gettime(CLOCK_MONOTONIC, &ice_start);

    const int64_t N = 1000000000;
    const double h = 1.0 / N;

    double sum = 0.0;
    #pragma omp parallel for reduction(+:sum)
    for (int64_t i = 0; i < N; ++i) {
        double x = (i + 0.5) * h;
        sum += 4.0 / (1.0 + x * x);
    }

    printf("pi = %.15f with %d threads\n", sum * h, omp_get_max_threads());

    clock_gettime(CLOCK_MONOTONIC, &ice_end);
    printf("Total time: %.3f ms\n", 1e3 * (ice_end.tv_sec - ice_start.tv_sec) + 1e-6 * (ice_end.tv_nsec - ice_start.tv_nsec));
    return 0;
}
