#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <math.h>
#include <unistd.h>
#include <time.h>

typedef struct
{
    int thread_id;
    int num_threads;
    int rounds;
    volatile int **my_flags;
    volatile int **partner_flags; // Change from *** to **
} ThreadData;

volatile int ***flags; // 3D array for all threads and rounds

void *disseminationBarrier(void *arg)
{
    ThreadData *data = (ThreadData *)arg;
    int num_threads = data->num_threads;
    int rounds = data->rounds;
    int thread_id = data->thread_id;

    int sense = 1;

    for (int barrier = 0; barrier < 1000; barrier++) {
        for (int r = 0; r < rounds; r++) {
            // Signal partner
            __atomic_store_n(data->partner_flags[r], sense, __ATOMIC_RELEASE);

            // Wait for signal from partner
            while (__atomic_load_n(data->my_flags[r], __ATOMIC_ACQUIRE) != sense) {
                sched_yield();
            }
        }
        // Reset my flags for the next barrier
        for (int r = 0; r < rounds; r++) {
            __atomic_store_n(data->my_flags[r], 1 - sense, __ATOMIC_RELEASE);
        }

        if (barrier % 100 == 0)
            printf("Thread %d reached barrier %d\n", data->thread_id, barrier);

        sense = 1 - sense;
    }


    return NULL;
}

int main(int argc, char* argv[])
{

    int num_threads = atoi(argv[1]);
    int rounds = (int)ceil(log2(num_threads));

    pthread_t *threads = malloc(num_threads * sizeof(pthread_t));
    ThreadData *thread_data = malloc(num_threads * sizeof(ThreadData));

    // Allocate 3D flag array: flags[thread][round][0]
    flags = malloc(num_threads * sizeof(volatile int **));
    for (int i = 0; i < num_threads; i++)
    {
        flags[i] = malloc(rounds * sizeof(volatile int *));
        for (int r = 0; r < rounds; r++)
        {
            // flags[i][r] = malloc(sizeof(volatile int));
            // *flags[i][r] = 0;
            int *ptr = malloc(sizeof(int));
            __atomic_store_n(ptr, 0, __ATOMIC_RELEASE);
            flags[i][r] = ptr;
        }
    }

    for (int i = 0; i < num_threads; i++)
    {
        thread_data[i].thread_id = i;
        thread_data[i].num_threads = num_threads;
        thread_data[i].rounds = rounds;

        thread_data[i].my_flags = malloc(rounds * sizeof(volatile int *));
        thread_data[i].partner_flags = malloc(rounds * sizeof(volatile int *)); // <-- FIXED

        for (int r = 0; r < rounds; r++)
        {
            int partner = (i + (1 << r)) % num_threads;
            printf("Thread %d round %d partner: %d\n", i, r, partner);

            thread_data[i].my_flags[r] = flags[i][r];
            thread_data[i].partner_flags[r] = flags[partner][r];
        }
    }

    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    for (int i = 0; i < num_threads; i++)
    {
        pthread_create(&threads[i], NULL, disseminationBarrier, &thread_data[i]);
    }

    for (int i = 0; i < num_threads; i++)
    {
        pthread_join(threads[i], NULL);
    }

    clock_gettime(CLOCK_MONOTONIC, &end);

    double elapsed_sec = end.tv_sec - start.tv_sec;
    double elapsed_nsec = end.tv_nsec - start.tv_nsec;
    double total_time = elapsed_sec + elapsed_nsec / 1e9;
    double avg_latency_us = (total_time / 1000) * 1e6; // microseconds per barrier

    printf("Average barrier latency: %.3f microseconds\n", avg_latency_us);

    // Free allocated memory
    for (int i = 0; i < num_threads; i++)
    {
        for (int r = 0; r < rounds; r++)
        {
            free((void *)flags[i][r]);
        }
        free((void *)flags[i]);

        free((void *)thread_data[i].my_flags);
        free((void *)thread_data[i].partner_flags);
    }
    free(flags);
    free(thread_data);
    free(threads);
}