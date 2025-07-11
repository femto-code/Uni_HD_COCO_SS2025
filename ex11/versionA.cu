#include <stdio.h>
#include <cuda_runtime.h>

#define N 256

// Device globals
__device__ int data[N];
__device__ int flag = 0;
__device__ int result = 0;

__global__ void producer_consumer_baseline() {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        // Producer writes to data and sets flag
        for (int i = 0; i < N; ++i)
            data[i] = i * 10;
        flag = 1; // No threadfence
    }

    if (blockIdx.x == 1 && threadIdx.x == 0) {
        // Consumer spins on flag
        while (flag == 0);

        int sum = 0;
        for (int i = 0; i < N; ++i)
            sum += data[i];

        result = sum;
    }
}

int main() {
    int expected = (N - 1) * N / 2 * 10;
    int host_result;
    int zero = 0;
    int errors = 0;

    for (int trial = 0; trial < 1000; ++trial) {
        // Reset device memory (flag and result)
        cudaMemcpyToSymbol(flag, &zero, sizeof(int));
        cudaMemcpyToSymbol(result, &zero, sizeof(int));

        // Launch kernel: 2 blocks, 1 thread each
        producer_consumer_baseline<<<2, 1>>>();
        cudaDeviceSynchronize();

        // Read result back
        cudaMemcpyFromSymbol(&host_result, result, sizeof(int));

        if (host_result != expected) {
            printf("Trial %d failed! Got %d, expected %d\n", trial, host_result, expected);
            errors++;
        }
    }

    if (errors == 0)
        printf("All runs succeeded.\n");
    else
        printf("%d runs failed.\n", errors);

    return 0;
}
