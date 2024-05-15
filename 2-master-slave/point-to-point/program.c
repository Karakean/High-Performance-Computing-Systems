#include "utility.h"
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <mpi.h>
#include "numgen.c"

#define WORK_REQUEST 1
#define WORK_SEND 2
#define WORK_DONE 3
#define NO_MORE_WORK 4

int is_prime(unsigned long int number) {
    if (number <= 1) return 0;
    if (number <= 3) return 1;
    if (number % 2 == 0 || number % 3 == 0) return 0;
    for (unsigned long int i = 5; i * i <= number; i += 6) {
        if (number % i == 0 || number % (i + 2) == 0) return 0;
    }
    return 1;
}

int main(int argc, char **argv) {
    struct timeval ins__tstart, ins__tstop;
    MPI_Init(&argc, &argv);
    int myrank, nproc;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);

    Args ins__args;
    parseArgs(&ins__args, &argc, argv);
    long inputArgument = ins__args.arg;
    long chunkSize = inputArgument / (10 * nproc);
    long currentIndex = 0;

    if (myrank == 0) { // Master process
        unsigned long int *numbers = (unsigned long int*)malloc(inputArgument * sizeof(unsigned long int));
        numgen(inputArgument, numbers);
        gettimeofday(&ins__tstart, NULL);
        int activeWorkers = nproc - 1;
        int totalPrimes = 0;
        while (activeWorkers > 0) {
            MPI_Status status;
            int count = 0;
            MPI_Recv(&count, 1, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
            if (status.MPI_TAG == WORK_REQUEST) {
                if (currentIndex < inputArgument) {
                    long sendCount = (currentIndex + chunkSize > inputArgument) ? inputArgument - currentIndex : chunkSize;
                    MPI_Send(numbers + currentIndex, sendCount, MPI_UNSIGNED_LONG, status.MPI_SOURCE, WORK_SEND, MPI_COMM_WORLD);
                    currentIndex += sendCount;
                } else {
                    MPI_Send(NULL, 0, MPI_UNSIGNED_LONG, status.MPI_SOURCE, NO_MORE_WORK, MPI_COMM_WORLD);
                    activeWorkers--;
                }
            }
            totalPrimes += count;
        }
        printf("Total number of primes for %ld elements: %d\n", inputArgument, totalPrimes);
        gettimeofday(&ins__tstop, NULL);
        ins__printtime(&ins__tstart, &ins__tstop, ins__args.marker);
        free(numbers);
    } else { // Worker processes
        while (1) {
            int count = 0;
            MPI_Send(&count, 1, MPI_INT, 0, WORK_REQUEST, MPI_COMM_WORLD);
            unsigned long int *recvbuf = (unsigned long int*)malloc(chunkSize * sizeof(unsigned long int));
            MPI_Status status;
            MPI_Recv(recvbuf, chunkSize, MPI_UNSIGNED_LONG, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
            if (status.MPI_TAG == NO_MORE_WORK) {
                free(recvbuf);
                break;
            }
            int numReceived;
            MPI_Get_count(&status, MPI_UNSIGNED_LONG, &numReceived);
            int primecount = 0;
            for (int i = 0; i < numReceived; i++) {
                if (is_prime(recvbuf[i])) {
                    primecount++;
                }
            }
            MPI_Send(&primecount, 1, MPI_INT, 0, WORK_DONE, MPI_COMM_WORLD);
            free(recvbuf);
        }
    }

    MPI_Finalize();
    return 0;
}
