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

    int myrank, nproc;
    unsigned long int *numbers;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);

    int *sendcounts = malloc(nproc * sizeof(int));
    int *displs = malloc(nproc * sizeof(int));
    int sum = 0;
    for (int i = 0; i < nproc; i++) {
        sendcounts[i] = inputArgument / nproc;
        if (i == nproc - 1) {
            sendcounts[i] += inputArgument % nproc;
        }
        displs[i] = sum;
        sum += sendcounts[i];
    }

    unsigned long int *recvbuf = (unsigned long int*)malloc(sendcounts[myrank] * sizeof(unsigned long int));
    MPI_Request scatter_req, gather_req;

    if (!myrank) {
        gettimeofday(&ins__tstart, NULL);
        numbers = (unsigned long int*)malloc(inputArgument * sizeof(unsigned long int));
        numgen(inputArgument, numbers);
    }

    MPI_Iscatterv(numbers, sendcounts, displs, MPI_UNSIGNED_LONG, recvbuf, sendcounts[myrank], MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD, &scatter_req);
    MPI_Wait(&scatter_req, MPI_STATUS_IGNORE); // de facto blocking

    int primecount = 0;
    for (int i = 0; i < sendcounts[myrank]; i++) {
        if (is_prime(recvbuf[i])) {
            primecount++;
        }
    }

    int *all_primecounts = NULL;
    if (!myrank) {
        all_primecounts = malloc(nproc * sizeof(int));
    }
    MPI_Igather(&primecount, 1, MPI_INT, all_primecounts, 1, MPI_INT, 0, MPI_COMM_WORLD, &gather_req);
    MPI_Wait(&gather_req, MPI_STATUS_IGNORE); // de facto blocking

    if (!myrank) {
        int total_primes = 0;
        for (int i = 0; i < nproc; i++) {
            total_primes += all_primecounts[i];
        }
        printf("Total number of primes for %ld elements: %d\n", inputArgument, total_primes);
        gettimeofday(&ins__tstop, NULL);
        ins__printtime(&ins__tstart, &ins__tstop, ins__args.marker);
        free(numbers);
        free(all_primecounts);
    }

    free(recvbuf);
    free(sendcounts);
    free(displs);

    MPI_Finalize();
    return 0;
}
