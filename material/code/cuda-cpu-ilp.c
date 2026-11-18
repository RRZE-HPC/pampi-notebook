#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// wall-clock time in seconds; timespec_get() is standard C11, unlike POSIX clock_gettime()
static double seconds(void) {
    struct timespec t;
    timespec_get(&t, TIME_UTC);
    return t.tv_sec + 1e-9 * t.tv_nsec;
}

// the sum of a, reps times, with 1 to 8 independent partial sums (dependency chains):
// an addition in one chain does not need the results of the other chains.
// Inlined with a constant number of chains, so that the partial sums stay in registers.
static inline __attribute__((always_inline)) double sum(const double *a, int n, int reps, int chains) {
    double p[8] = { 0.0 };
    for (int r = 0; r < reps; ++r)
        for (int i = 0; i < n; i += chains)
            for (int k = 0; k < chains; ++k)
                p[k] += a[i + k];
    return p[0] + p[1] + p[2] + p[3] + p[4] + p[5] + p[6] + p[7];
}

int main(void) {
    const int n = 2048, reps = 100000;          // 16 KiB: stays in the L1 cache, no memory latency involved
    double *a = malloc(n * sizeof(double));
    for (int i = 0; i < n; ++i)
        a[i] = 1.0 / (i + 1);

    printf("chains  ns per addition  sum\n");
    for (int chains = 1; chains <= 8; chains *= 2) {
        double t0 = seconds(), s;
        switch (chains) {
            case 1:  s = sum(a, n, reps, 1); break;     // every addition waits for the previous one
            case 2:  s = sum(a, n, reps, 2); break;
            case 4:  s = sum(a, n, reps, 4); break;
            default: s = sum(a, n, reps, 8);            // up to eight additions in flight
        }
        double t = seconds() - t0;
        printf("%6d  %15.3f  %.10f\n", chains, 1e9 * t / ((double)n * reps), s);
    }
    free(a);
    return 0;
}
