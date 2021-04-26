CUCOMP = nvcc
CFLAGS =

run: vector_addition.o
	$(CUCOMP) $(CFLAGS) vector_addition.o -o run

vector_addition.o: vector_addition.cu
	$(CUCOMP) $(CFLAGS) -c vector_addition.cu

.PHONY: clean cleanall

clean:
	rm -f run *.o

cleanall:
	rm -f run *.o add_vec*
