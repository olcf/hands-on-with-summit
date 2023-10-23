CCOMP = mpicc
CFLAGS = -fopenmp
PREP = 

run: jacobi.o main.o
	$(PREP) $(CCOMP) $(CFLAGS) jacobi.o main.o -o run
main.o: main.c jacobi.c jacobi.h
	$(PREP) $(CCOMP) $(CFLAGS) -c main.c

jacobi.o: jacobi.c jacobi.h
	$(PREP) $(CCOMP) $(CFLAGS) -c jacobi.c

.PHONY: clean cleanall

clean: 
	rm -f *.o run*

cleanall:
	rm -f *.o run jacobi*
