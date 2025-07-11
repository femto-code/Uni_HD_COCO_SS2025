#include <stdio.h>
#include <cuda_runtime.h>

#define N 32

__device__ int result = 0;
__device__ unsigned long long consumer_time = 0;

__global__ void versionBonus() {
    int tid = threadIdx.x;
    int value = tid * 10;

    // Producer: thread 0
    if (tid == 0) {
        value = 12345; // Example message
    }

    // Broadcast value from thread 0 to all threads in the warp
    int msg = __shfl_sync(0xFFFFFFFF, value, 0);

    // Consumer: all threads receive the message
    unsigned long long start = clock64();
    int sum = msg; // For demonstration, sum is just the message
    unsigned long long end = clock64();

    if (tid == 0) {
        result = sum;
        consumer_time = end - start;
    }
}

int main() {
    int host_result = 0;
    int zero = 0;

    cudaMemcpyToSymbol(result, &zero, sizeof(int));
    cudaMemcpyToSymbol(consumer_time, &zero, sizeof(unsigned long long));

    versionBonus<<<1, N>>>();
    cudaDeviceSynchronize();

    cudaMemcpyFromSymbol(&host_result, result, sizeof(int));
    unsigned long long host_consumer_time = 0;
    cudaMemcpyFromSymbol(&host_consumer_time, consumer_time, sizeof(unsigned long long));

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    float consumer_ms = (float)host_consumer_time / prop.clockRate * 1000.0f;

    printf("Bonus (warp shuffle): result: %d, consumer loop: %.6f ms\n", host_result, consumer_ms);

    return 0;
}