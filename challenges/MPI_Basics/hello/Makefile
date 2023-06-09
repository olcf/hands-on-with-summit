CCOMP  = cc
CFLAGS =

run: mpi_hello.o
	$(CCOMP) $(CFLAGS) mpi_hello.o -o run

mpi_hello.o: mpi_hello.c
	$(CCOMP) $(CFLAGS) -c mpi_hello.c

.PHONY: clean cleanall

clean:
	rm -f run *.o

cleanall:
	rm -f run *.o hello*
