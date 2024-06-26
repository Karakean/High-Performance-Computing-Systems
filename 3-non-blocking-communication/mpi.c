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

int main(int argc, char **argv) {

    Args ins__args;
    parseArgs(&ins__args, &argc, argv);
    long inputArgument = ins__args.arg;
    struct timeval ins__tstart, ins__tstop;

    MPI_Init(&argc, &argv);
    int myrank, nproc;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);

    unsigned long int *numbers = NULL;
    int *sendcounts = malloc(nproc * sizeof(int));
    int *displs = malloc(nproc * sizeof(int));

    MPI_Request *send_requests = malloc(nproc * sizeof(MPI_Request));
    MPI_Request *recv_requests = malloc(nproc * sizeof(MPI_Request));
    MPI_Status *statuses = malloc(nproc * sizeof(MPI_Status));

    int sum = 0;
    for (int i = 0; i < nproc; i++) {
        sendcounts[i] = inputArgument / nproc;
        if (i == nproc - 1) {
            sendcounts[i] += inputArgument % nproc;
        }
        displs[i] = sum;
        sum += sendcounts[i];
    }

    unsigned long int *recvbuf;

    if (!myrank) { // root
        numbers = (unsigned long int *)malloc(inputArgument * sizeof(unsigned long int));
        numgen(inputArgument, numbers);
        gettimeofday(&ins__tstart, NULL);

        recvbuf = numbers;

        for (int i = 1; i < nproc; i++) {
            MPI_Isend(numbers + displs[i], sendcounts[i], MPI_UNSIGNED_LONG, i, 0, MPI_COMM_WORLD, &send_requests[i-1]);
        }
    } else {
        recvbuf = (unsigned long int *)malloc(sendcounts[myrank] * sizeof(unsigned long int));
        MPI_Irecv(recvbuf, sendcounts[myrank], MPI_UNSIGNED_LONG, 0, 0, MPI_COMM_WORLD, &recv_requests[myrank-1]);
        MPI_Wait(&recv_requests[myrank-1], MPI_STATUS_IGNORE);
    }

    int primecount = 0;
    for (int i = 0; i < sendcounts[myrank]; i++) {
        if (is_prime(recvbuf[i])) {
            primecount++;
        }
    }

    int *all_prime_counts = NULL;
    if (myrank == 0) {
        all_prime_counts = malloc(nproc * sizeof(int));
    }

    if (myrank == 0) {
        all_prime_counts[0] = primecount;
        for (int i = 1; i < nproc; i++) {
            MPI_Irecv(&all_prime_counts[i], 1, MPI_INT, i, 0, MPI_COMM_WORLD, &recv_requests[i-1]);
        }
    } else {
        MPI_Isend(&primecount, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &send_requests[myrank-1]);
    }

    if (myrank == 0) {
        MPI_Waitall(nproc - 1, recv_requests, statuses);
    }

    if (myrank == 0) {
        int total_primes = 0;
        for (int i = 0; i < nproc; i++) {
            total_primes += all_prime_counts[i];
        }
        printf("Total number of primes for %ld elements: %d\n", inputArgument, total_primes);
        gettimeofday(&ins__tstop, NULL);
        ins__printtime(&ins__tstart, &ins__tstop, ins__args.marker);
        free(numbers);
        free(all_prime_counts);
    } else {
        free(recvbuf);
    }

    free(sendcounts);
    free(displs);
    free(send_requests);
    free(recv_requests);
    free(statuses);

    MPI_Finalize();
    return 0;
}
