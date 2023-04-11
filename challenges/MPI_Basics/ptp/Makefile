CCOMP = cc
CFLAGS =

run: ptp.o
	$(CCOMP) $(CFLAGS) ptp.o -o run

ptp.o: ptp.c
	$(CCOMP) $(CFLAGS) -c ptp.c

.PHONY: clean cleanall

clean:
	rm -f run *.o

cleanall:
	rm -f run *.o hello* 
