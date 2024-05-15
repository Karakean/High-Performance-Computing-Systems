MPICC=mpicc
CFLAGS=-O2
LIBS=-lm
OMPFLAGS=-fopenmp

ifeq (run,$(firstword $(MAKECMDGOALS)))
  # use the rest as arguments for "run"
  RUN_ARGS := $(wordlist 2,$(words $(MAKECMDGOALS)),$(MAKECMDGOALS))
  # ...and turn them into do-nothing targets
  $(eval $(RUN_ARGS):;@:)
endif


main: program.c
	${MPICC} ${CFLAGS} ${OMPFLAGS} $< -o program.out ${LIBS}

run:
	mpirun -bind-to none -np 2 ./program.out $(RUN_ARGS)

clean:
	rm program.out
