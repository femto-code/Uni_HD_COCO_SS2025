#include <stdio.h>
#include <cuda_runtime.h>

#define N 256

__device__ int data[N];
__device__ volatile int flag = 0;
__device__ int result = 0;

__device__ unsigned long long consumer_time = 0;

__global__ void versionB1() {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        for (int i = 0; i < N; ++i)
            data[i] = i * 10;

        __threadfence();  // Ensure data[] is visible before flag
        flag = 1;
    }

    if (blockIdx.x == 1 && threadIdx.x == 0) {
        while (flag == 0);

        // Start timing
        unsigned long long start = clock64();

        int sum = 0;
        for (int i = 0; i < N; ++i)
            sum += data[i];

        // End timing
        unsigned long long end = clock64();
        consumer_time = end - start;

        result = sum;
    }
}

__global__ void versionB2() {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        for (int i = 0; i < N; ++i)
            data[i] = i * 10;

        flag = 1;
    }

    if (blockIdx.x == 1 && threadIdx.x == 0) {
        while (flag == 0);

        __threadfence();

        // Start timing
        unsigned long long start = clock64();

        int sum = 0;
        for (int i = 0; i < N; ++i)
            sum += data[i];

        // End timing
        unsigned long long end = clock64();
        consumer_time = end - start;

        result = sum;
    }
}

__global__ void versionB3() {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        for (int i = 0; i < N; ++i)
            data[i] = i * 10;

        __threadfence();
        flag = 1;
    }

    if (blockIdx.x == 1 && threadIdx.x == 0) {
        while (flag == 0);

        __threadfence();

        // Start timing
        unsigned long long start = clock64();

        int sum = 0;
        for (int i = 0; i < N; ++i)
            sum += data[i];

        // End timing
        unsigned long long end = clock64();
        consumer_time = end - start;

        result = sum;
    }
}

void run_version(const char* label, void (*kernel)()) {
    int host_result = 0;
    int zero = 0;
    int expected = (N - 1) * N / 2 * 10;

    // Reset
    cudaMemcpyToSymbol(flag, &zero, sizeof(int));
    cudaMemcpyToSymbol(result, &zero, sizeof(int));
    cudaMemcpyToSymbol(consumer_time, &zero, sizeof(unsigned long long));

    // Timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    kernel<<<2, 1>>>();
    cudaEventRecord(stop);
    cudaDeviceSynchronize();

    cudaMemcpyFromSymbol(&host_result, result, sizeof(int));
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);

    unsigned long long host_consumer_time = 0;
    cudaMemcpyFromSymbol(&host_consumer_time, consumer_time, sizeof(unsigned long long));

    // Convert cycles to ms (approximate, using clock rate)
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    float consumer_ms = (float)host_consumer_time / prop.clockRate * 1000.0f;

    printf("%s => result: %d [%s], total time: %.3f ms, consumer loop: %.6f ms\n",
       label, host_result,
       (host_result == expected ? "OK" : "FAIL"), ms, consumer_ms);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main() {
    printf("=== Version B: Memory Fence Variants ===\n");

    run_version("Version B1: fence in producer", versionB1);
    run_version("Version B2: fence in consumer", versionB2);
    run_version("Version B3: fence in both", versionB3);

    return 0;
}
