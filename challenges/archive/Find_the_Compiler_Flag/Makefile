CCOMP = gcc
CFLAGS =

run: vector_addition.o
	$(CCOMP) $(CFLAGS) vector_addition.o -o run

vector_addition.o: vector_addition.c
	$(CCOMP) $(CFLAGS) -c vector_addition.c

.PHONY: clean cleanall

clean:
	rm -f run *.o

cleanall:
	rm -f run *.o add_vec*
