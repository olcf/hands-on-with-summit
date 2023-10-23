COMP = mpicc
CFLAGS =
PREP = scorep

run: jacobi.o main.o
	$(PREP) $(COMP) $(CFLAGS) main.o jacobi.o -o run

main.o: main.c jacobi.c jacobi.h
	$(PREP) $(COMP) $(CFLAGS) -c main.c

jacobi.o: jacobi.c jacobi.h
	$(PREP) $(COMP) $(CFLAGS) -c jacobi.c

.PHONY: clean cleanall

clean: 
	rm -f *.o run*

cleanall:
	rm -f *.o run jacobi*
