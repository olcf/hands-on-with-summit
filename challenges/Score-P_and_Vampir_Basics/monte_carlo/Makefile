CCOMP  = mpicc
CFLAGS =

run: calculating_pi.o
	$(CCOMP) $(CFLAGS) calculating_pi.o -o run

calculating_pi.o: calcPiMPI.c
	$(CCOMP) $(CFLAGS) -c calculating_pi.c

.PHONY: clean cleanall

clean:
	rm -f run *.o

cleanall:
	rm -f run *.o monte_carlo*
