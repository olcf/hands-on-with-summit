COMP  = cc
FLAGS = -fopenmp

hello_mpi_omp: hello_mpi_omp.o
        ${COMP} ${FLAGS} hello_mpi_omp.o -o hello_mpi_omp

hello_mpi_omp.o: hello_mpi_omp.c
        ${COMP} ${FLAGS} -c hello_mpi_omp.c

PHONY: clean

clean:
        rm -f hello_mpi_omp *.o
