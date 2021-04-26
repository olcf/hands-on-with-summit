CCOMP = gcc
CFLAGS = -fopenmp

hello: hello_world.o
	$(CCOMP) $(CFLAGS) hello_world.o -o hello

hello_world.o: hello_world.c
	$(CCOMP) $(CFLAGS) -c hello_world.c

.PHONY: clean cleanall

clean:
	rm -f hello *.o

cleanall:
	rm -f hello *.o hello_test*
