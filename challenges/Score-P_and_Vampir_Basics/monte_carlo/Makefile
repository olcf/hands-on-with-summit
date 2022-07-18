CCOMP  = mpicc
CFLAGS =

run: calcPiMPI.o
        $(CCOMP) $(CFLAGS) calcPiMPI.o -o run

calcPiMPI.o: calcPiMPI.c
        $(CCOMP) $(CFLAGS) -c calcPiMPI.c

.PHONY: clean cleanall

clean:
        rm -f run *.o

cleanall:
        rm -f run *.o monte_carlo*
