CCOMP = gcc
CFLAGS = -fopenmp

vec_add: vector_addition.o
	$(CCOMP) $(CFLAGS) vector_addition.o -o vec_add

vector_addition.o: vector_addition.c
	$(CCOMP) $(CFLAGS) -c vector_addition.c

.PHONY: clean cleanall

clean:
	rm -f vec_add *.o

cleanall:
	rm -f vec_add *.o vec_add_test*
