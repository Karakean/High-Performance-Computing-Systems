#include "utility.h"
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <mpi.h>
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

int main(int argc,char **argv) {

    Args ins__args;
    parseArgs(&ins__args, &argc, argv);
    long inputArgument = ins__args.arg; 
    struct timeval ins__tstart, ins__tstop;

    MPI_Init(&argc,&argv);
    int myrank, nproc;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);

    unsigned long int *numbers = NULL;
    int *sendcounts = malloc(nproc * sizeof(int));
    int *displs = malloc(nproc * sizeof(int));

    
    for (int i = 0; i < nproc; i++) {
        int sum = 0; 
        sendcounts[i] = inputArgument / nproc;
        if (i == nproc - 1) {
            sendcounts[i] += inputArgument % nproc; // Add possible remainder to the last process
        }
        displs[i] = sum;
        sum += sendcounts[i];
    }

    unsigned long int *recvbuf = (unsigned long int*)malloc(sendcounts[myrank] * sizeof(unsigned long int));

    if (!myrank) { // root
        numbers = (unsigned long int*)malloc(inputArgument * sizeof(unsigned long int));
        numgen(inputArgument, numbers);
        gettimeofday(&ins__tstart, NULL);
        memcpy(recvbuf, numbers, sendcounts[0] * sizeof(unsigned long int));
        for (int i = 1; i < nproc; i++) {
            MPI_Send(numbers + displs[i], sendcounts[i], MPI_UNSIGNED_LONG, i, 0, MPI_COMM_WORLD);
        }
    } else {
        MPI_Recv(recvbuf, sendcounts[myrank], MPI_UNSIGNED_LONG, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    int primecount = 0;
    for (int i = 0; i < sendcounts[myrank]; i++) {
        if (is_prime(recvbuf[i])) {
            primecount++;
        }
    }

    if (myrank == 0) {
        int total_primes = primecount;
        int other_prime_count;
        for (int i = 1; i < nproc; i++) {
            MPI_Recv(&other_prime_count, 1, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            total_primes += other_prime_count;
        }
        printf("Total number of primes for %ld elements: %d\n", inputArgument, total_primes);
        gettimeofday(&ins__tstop, NULL);
        ins__printtime(&ins__tstart, &ins__tstop, ins__args.marker);
        free(numbers);
    } else {
        MPI_Send(&primecount, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
    }

    free(recvbuf);
    free(sendcounts);
    free(displs);

    MPI_Finalize();
    return 0;
}
