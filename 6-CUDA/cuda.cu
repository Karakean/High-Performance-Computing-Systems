#include "utility.h"
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <sys/time.h>
#include "numgen.c"

int is_prime(unsigned long int number) {
    if (number <= 1) return 0;
    if (number <= 3) return 1;
    if (number % 2 == 0 || number % 3 == 0) return 0;
    for (unsigned long int i = 5; i * i <= number; i += 6) {
        if (number % i == 0 || number % (i + 2) == 0) return 0;
    }
    return 1;
}

__global__ void check_primes(unsigned long int *numbers, int *prime_counts, long inputArgument) {
    long idx = blockIdx.x * blockDim.x + threadIdx.x;
    long stride = gridDim.x * blockDim.x;
    int local_prime_count = 0;
    for (long i = idx; i < inputArgument; i += stride) {
        if (is_prime(numbers[i])) {
            local_prime_count++;
        }
    }
    atomicAdd(prime_counts, local_prime_count);
}

int main(int argc, char **argv) {
    Args ins__args;
    parseArgs(&ins__args, &argc, argv);

    long inputArgument = ins__args.arg;
    unsigned long int *numbers = (unsigned long int*)malloc(inputArgument * sizeof(unsigned long int));
    numgen(inputArgument, numbers);

    unsigned long int *d_numbers;
    int *d_prime_counts;
    cudaMalloc((void**)&d_numbers, inputArgument * sizeof(unsigned long int));
    cudaMalloc((void**)&d_prime_counts, sizeof(int));

    cudaMemcpy(d_numbers, numbers, inputArgument * sizeof(unsigned long int), cudaMemcpyHostToDevice);

    int zero = 0;
    cudaMemcpy(d_prime_counts, &zero, sizeof(int), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int gridSize = (inputArgument + blockSize - 1) / blockSize;

    struct timeval ins__tstart, ins__tstop;
    gettimeofday(&ins__tstart, NULL);

    check_primes<<<gridSize, blockSize>>>(d_numbers, d_prime_counts, inputArgument);
    cudaDeviceSynchronize();

    gettimeofday(&ins__tstop, NULL);
    ins__printtime(&ins__tstart, &ins__tstop, ins__args.marker);

    int prime_count;
    cudaMemcpy(&prime_count, d_prime_counts, sizeof(int), cudaMemcpyDeviceToHost);

    printf("Total number of prime numbers: %d\n", prime_count);

    free(numbers);
    cudaFree(d_numbers);
    cudaFree(d_prime_counts);

    return 0;
}