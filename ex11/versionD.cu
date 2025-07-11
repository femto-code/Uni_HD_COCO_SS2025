#include <stdio.h>
#include <cuda_runtime.h>

#define N 256

__device__ int result = 0;
__device__ unsigned long long consumer_time = 0;

__global__ void versionD() {
    __shared__ int data[N];

    int tid = threadIdx.x;

    // Producer: first half of threads
    if (tid < N / 2) {
        for (int i = tid; i < N; i += N / 2)
            data[i] = i * 10;
    }

    __syncthreads(); // Ensure all writes to shared memory are visible

    // Consumer: second half of threads
    if (tid >= N / 2) {
        unsigned long long start = clock64();
        int sum = 0;
        for (int i = 0; i < N; ++i)
            sum += data[i];
        unsigned long long end = clock64();

        if (tid == N / 2) { // Only one thread writes result
            result = sum;
            consumer_time = end - start;
        }
    }
    __syncthreads(); // Optional: ensure all consumers are done
}

int main() {
    int host_result = 0;
    int zero = 0;

    cudaMemcpyToSymbol(result, &zero, sizeof(int));
    cudaMemcpyToSymbol(consumer_time, &zero, sizeof(unsigned long long));

    // Launch one block, N threads
    versionD<<<1, N>>>();
    cudaDeviceSynchronize();

    cudaMemcpyFromSymbol(&host_result, result, sizeof(int));
    unsigned long long host_consumer_time = 0;
    cudaMemcpyFromSymbol(&host_consumer_time, consumer_time, sizeof(unsigned long long));

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    float consumer_ms = (float)host_consumer_time / prop.clockRate * 1000.0f;

    int expected = (N - 1) * N / 2 * 10;
    printf("Version D (shared memory): result: %d [%s], consumer loop: %.6f ms\n",
           host_result, (host_result == expected ? "OK" : "FAIL"), consumer_ms);

    return 0;
}