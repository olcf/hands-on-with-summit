HIPCC = hipcc
CFLAGS = 

run: vector_addition.o
	$(HIPCC) $(CFLAGS) vector_addition.o -o run

vector_addition.o: vector_addition.cpp
	$(HIPCC) $(CFLAGS) -c vector_addition.cpp

.PHONY: clean cleanall

clean:
	rm -f run *.o

cleanall:
	rm -f run *.o add_vec*
