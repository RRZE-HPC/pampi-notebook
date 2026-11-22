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
    const uint32_t n = 1u << 25;                // 128 MiB: much larger than the caches
    const int64_t steps = 1 << 23;

    // a random-looking cycle through all n elements: next[i] = (a * i + c) mod n is a single cycle
    // for n = 2^k, odd c and a = 1 mod 4 (a full-period linear congruential generator)
    uint32_t *next = malloc(n * sizeof(uint32_t));
    for (uint32_t i = 0; i < n; ++i)
        next[i] = (1664525u * i + 1013904223u) & (n - 1);

    // the addresses visited by the chain, recorded in advance
    uint32_t *addr = malloc(steps * sizeof(uint32_t)), p = 0;
    for (int64_t s = 0; s < steps; ++s) {
        addr[s] = p;
        p = next[p];
    }

    // dependent loads: the address of the next load is the result of this one
    double t0 = seconds();
    p = 0;
    for (int64_t s = 0; s < steps; ++s)
        p = next[p];
    double t1 = seconds();

    // the same loads, but independent: all addresses are known in advance
    uint64_t sum = 0;
    for (int64_t s = 0; s < steps; ++s)
        sum += next[addr[s]];
    double t2 = seconds();

    printf("dependent loads  : %6.2f ns per load (p = %u)\n", 1e9 * (t1 - t0) / steps, p);
    printf("independent loads: %6.2f ns per load (sum = %llu)\n", 1e9 * (t2 - t1) / steps, (unsigned long long)sum);
    free(next);
    free(addr);
    return 0;
}
