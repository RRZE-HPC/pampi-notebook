#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>

// wall-clock time in seconds; timespec_get() is standard C11, unlike POSIX clock_gettime()
static double seconds(void) {
    struct timespec t;
    timespec_get(&t, TIME_UTC);
    return t.tv_sec + 1e-9 * t.tv_nsec;
}

int main(void) {
    const int64_t n = 1 << 25;                  // the same size as on the GPU: 256 MiB per array
    const size_t bytes = n * sizeof(double);
    double *a = malloc(bytes), *b = malloc(bytes), *c = malloc(bytes);
    for (int64_t i = 0; i < n; ++i) {
        a[i] = 0.0;
        b[i] = 1.0;
        c[i] = 2.0;
    }

    const int reps = 10;
    double t0 = seconds();
    for (int r = 0; r < reps; ++r) {
        for (int64_t i = 0; i < n; ++i)
            a[i] = b[i] + 3.0 * c[i];
        __asm__ volatile("" : : "r"(a) : "memory");     // keeps the compiler from dropping repetitions
    }
    double t = (seconds() - t0) / reps;

    printf("triad on one core: %.2f ms, %.1f GB/s (a[0] = %.1f)\n", 1e3 * t, 3 * bytes / t / 1e9, a[0]);
    free(a);
    free(b);
    free(c);
    return 0;
}
