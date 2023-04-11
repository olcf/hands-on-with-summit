CCOMP = cc
CFLAGS =

run: mpi_bcast.o
	$(CCOMP) $(CFLAGS) mpi_bcast.o -o run

mpi_bcast.o: mpi_bcast.c
	$(CCOMP) $(CFLAGS) -c mpi_bcast.c

.PHONY: clean cleanall

clean:
	rm -f run *.o

cleanall:
	rm -f run *.o bcast.*
