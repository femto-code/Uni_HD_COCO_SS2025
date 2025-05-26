#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <stdatomic.h>
#include <string.h>
#include <unistd.h>
#include <sys/time.h>
#include <math.h>

#define MAX_THREADS 64
#define NUM_BARRIERS 10000

// Barrier Type Enumration
// add new barier types here
typedef enum { BARRIER_PTHREAD, BARRIER_COUNTER, BARRIER_DISSEMINATION } barrier_type_t;

// Dissemination Barrier Implementation

typedef struct {
    int thread_id;
    int num_threads;
    int rounds;
    int sense; // thread-local sense
    volatile int ***my_flags;
    volatile int ***partner_flags;
} dissemination_thread_arg_t;

typedef struct {
    volatile int ****flags; // flags[thread][round][2] — 2 for double-buffering
    dissemination_thread_arg_t thread_args[MAX_THREADS];
} dissemination_barrier_t;

void dissemination_barrier_init(dissemination_barrier_t *barrier, int num_threads) {
    int rounds = (int)ceil(log2(num_threads));

    // Allocate 4D array: flags[thread][round][sense]
    barrier->flags = malloc(num_threads * sizeof(volatile int ***));
    for (int i = 0; i < num_threads; i++) {
        barrier->flags[i] = malloc(rounds * sizeof(volatile int **));
        for (int r = 0; r < rounds; r++) {
            barrier->flags[i][r] = malloc(2 * sizeof(volatile int *));
            for (int s = 0; s < 2; s++) {
                int *ptr = malloc(sizeof(int));
                *ptr = 0;
                barrier->flags[i][r][s] = ptr;
            }
        }
    }

    for (int i = 0; i < num_threads; i++) {
        dissemination_thread_arg_t *arg = &barrier->thread_args[i];
        arg->thread_id = i;
        arg->num_threads = num_threads;
        arg->rounds = rounds;
        arg->sense = 0;

        arg->my_flags = malloc(rounds * sizeof(volatile int **));
        arg->partner_flags = malloc(rounds * sizeof(volatile int **));
        for (int r = 0; r < rounds; r++) {
            int partner = (i + (1 << r)) % num_threads;
            arg->my_flags[r] = barrier->flags[i][r];
            arg->partner_flags[r] = barrier->flags[partner][r];
        }
    }
}

void dissemination_barrier_wait(dissemination_thread_arg_t *arg, int barrier_num) {
    int s = arg->sense;
    for (int r = 0; r < arg->rounds; r++) {
        // Signal partner
        __atomic_store_n(arg->partner_flags[r][s], 1, __ATOMIC_RELEASE);
        // Wait for partner to signal back
        while (__atomic_load_n(arg->my_flags[r][s], __ATOMIC_ACQUIRE) != 1) {
            sched_yield();
        }
    }
    // Prepare flags for next round
    for (int r = 0; r < arg->rounds; r++) {
        __atomic_store_n(arg->my_flags[r][s], 0, __ATOMIC_RELAXED);
    }
    arg->sense = 1 - s;
}

void dissemination_barrier_destroy(dissemination_barrier_t *barrier, int num_threads) {
    for (int i = 0; i < num_threads; i++) {
        for (int r = 0; r < barrier->thread_args[i].rounds; r++) {
            for (int s = 0; s < 2; s++) {
                free((void *)barrier->flags[i][r][s]);
            }
            free((void *)barrier->flags[i][r]);
        }
        free((void *)barrier->flags[i]);
        free((void *)barrier->thread_args[i].my_flags);
        free((void *)barrier->thread_args[i].partner_flags);
    }
    free(barrier->flags);
}


// Counter Barrier Implementation

typedef struct {
    atomic_int count;
    atomic_int sense;
    int num_threads;
} counter_barrier_t;

void counter_barrier_init(counter_barrier_t *barrier, int num_threads) {
    atomic_init(&barrier->count, 0);
    atomic_init(&barrier->sense, 0);
    barrier->num_threads = num_threads;
}

void counter_barrier_wait(counter_barrier_t *barrier, int *local_sense) {
    *local_sense = !(*local_sense);
    int pos = atomic_fetch_add(&barrier->count, 1);
    if (pos == barrier->num_threads - 1) {
        atomic_store(&barrier->count, 0);
        atomic_store(&barrier->sense, *local_sense);
    } else {
        while (atomic_load(&barrier->sense) != *local_sense) {
            // Optional: __asm__ __volatile__("pause");
        }
    }
}

// Thread Argument Structure

typedef struct {
    int thread_id;
    int num_threads;
    barrier_type_t type;
    counter_barrier_t *counter_barrier;
    pthread_barrier_t *pthread_barrier;
    dissemination_barrier_t *dissemination_barrier;
} thread_arg_t;


// Pthread Barrier Implementation

void* benchmark_thread_func(void *arg) {
    thread_arg_t *targ = (thread_arg_t*)arg;
    int local_sense = 0;
    dissemination_thread_arg_t *d_arg = NULL;

    if (targ->type == BARRIER_DISSEMINATION)
        d_arg = &targ->dissemination_barrier->thread_args[targ->thread_id];

    for (int i = 0; i < NUM_BARRIERS; i++) {
        switch (targ->type) {
            case BARRIER_PTHREAD:
                pthread_barrier_wait(targ->pthread_barrier);
                break;
            case BARRIER_COUNTER:
                counter_barrier_wait(targ->counter_barrier, &local_sense);
                break;
            case BARRIER_DISSEMINATION:
                dissemination_barrier_wait(d_arg, i);
                break;
        }
    }
    return NULL;
}

// Time helper

double get_time_sec() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + (tv.tv_usec / 1e6);
}

// Benchmark Function

void run_benchmark(int num_threads, barrier_type_t type) {
    if (type == BARRIER_DISSEMINATION && num_threads < 2) {
        printf("%s,%d,%.3f\n", "dissemination", num_threads, 0.0);
        return;
    }
    pthread_t threads[MAX_THREADS];
    thread_arg_t args[MAX_THREADS];
    counter_barrier_t counter_barrier;
    pthread_barrier_t pthread_barrier;
    dissemination_barrier_t dissemination_barrier;

    if (type == BARRIER_COUNTER) {
        counter_barrier_init(&counter_barrier, num_threads);
    } else if (type == BARRIER_PTHREAD) {
        pthread_barrier_init(&pthread_barrier, NULL, num_threads);
    } else if (type == BARRIER_DISSEMINATION) {
        dissemination_barrier_init(&dissemination_barrier, num_threads);
    }

    double start = get_time_sec();

    for (int i = 0; i < num_threads; i++) {
        args[i].thread_id = i;
        args[i].num_threads = num_threads;
        args[i].type = type;
        args[i].counter_barrier = &counter_barrier;
        args[i].pthread_barrier = &pthread_barrier;
        args[i].dissemination_barrier = &dissemination_barrier;
        pthread_create(&threads[i], NULL, benchmark_thread_func, &args[i]);
    }

    for (int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], NULL);
    }

    double end = get_time_sec();
    double total_time = end - start;
    double avg_latency = (total_time / NUM_BARRIERS) * 1e6; // microseconds

    const char *label = type == BARRIER_PTHREAD ? "pthread" :
                        type == BARRIER_COUNTER ? "counter" : "dissemination";
                        
    printf("%s,%d,%.2f\n", label, num_threads, avg_latency);

    if (type == BARRIER_PTHREAD) {
        pthread_barrier_destroy(&pthread_barrier);
    } else if (type == BARRIER_COUNTER) {
        // No specific destroy function for counter barrier
    } else if (type == BARRIER_DISSEMINATION) {
        dissemination_barrier_destroy(&dissemination_barrier, num_threads);
    }
}

// Main Function

int main(int argc, char *argv[]) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s pthread|counter|dissemination\\n", argv[0]);
        return EXIT_FAILURE;
    }

    barrier_type_t type;
    if (strcmp(argv[1], "pthread") == 0) {
        type = BARRIER_PTHREAD;
    } else if (strcmp(argv[1], "counter") == 0) {
        type = BARRIER_COUNTER;
    } else if (strcmp(argv[1], "dissemination") == 0) {
        type = BARRIER_DISSEMINATION;
    } else {
        fprintf(stderr, "Invalid barrier type.\n");
        return EXIT_FAILURE;
    }

    printf("barrier_type,num_threads,avg_latency_us\n");
    int thread_counts[] = {1, 2, 4, 8, 12, 16, 24, 32, 40, 48};
    int num_tests = sizeof(thread_counts) / sizeof(thread_counts[0]);

    for (int i = 0; i < num_tests; i++) {
        run_benchmark(thread_counts[i], type);
    }

    return 0;
}
