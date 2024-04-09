#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
 
int
main (int argc, char **argv)
{
    int myrank, proccount;
    double pi, pi_final;
    int chunk, rest, limit, mine, sign;
    int cardinality = argc > 1 ? atoi(argv[1]) : 10;
 
    // Initialize MPI
    MPI_Init (&argc, &argv);
 
    // find out my rank
    MPI_Comm_rank (MPI_COMM_WORLD, &myrank);
 
    // find out the number of processes in MPI_COMM_WORLD
    MPI_Comm_size (MPI_COMM_WORLD, &proccount);
 
    // each process performs computations on its part
    pi = 0;
    chunk = cardinality / proccount;
    rest = cardinality % proccount;
    limit = (myrank != (proccount - 1)) ? chunk : chunk + rest;
    mine = myrank * chunk * 2 + 1;
    sign = (((mine - 1) / 2) % 2) ? -1 : 1;

    for (int i = 0; i < limit; i++)
      {
        printf("Process %d %d %d\n", myrank, sign, mine);
        pi += sign / (double) mine;
        mine+=2;
        sign = (((mine - 1) / 2) % 2) ? -1 : 1;
      }
 
    // now merge the numbers to rank 0
    MPI_Reduce (&pi, &pi_final, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
 
    if (!myrank)
    {
      pi_final *= 4;
      printf ("pi=%.17g", pi_final);
    }
 
    // Shut down MPI
    MPI_Finalize ();
 
    return 0;
}
