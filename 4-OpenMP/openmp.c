#include "utility.h"
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <omp.h>
#include "numgen.c"


int main(int argc,char **argv) {
  Args ins__args;
  parseArgs(&ins__args, &argc, argv);
  omp_set_num_threads(ins__args.n_thr);
  long inputArgument = ins__args.arg; 
  unsigned long int *numbers = (unsigned long int*)malloc(inputArgument * sizeof(unsigned long int));
  numgen(inputArgument, numbers);
  struct timeval ins__tstart, ins__tstop;
  gettimeofday(&ins__tstart, NULL);
  
  int primeCount = 0;
  #pragma omp parallel for reduction(+:primeCount)
  for (int i = 0; i < inputArgument; i++) {
    primeCount += is_prime(numbers[i]);
  }
  
  printf("Primes number: %d", primeCount);

  gettimeofday(&ins__tstop, NULL);
  ins__printtime(&ins__tstart, &ins__tstop, ins__args.marker);

}
