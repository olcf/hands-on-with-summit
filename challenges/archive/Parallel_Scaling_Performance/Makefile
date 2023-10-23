CCOMP = cc

# CPU Threading
CFLAGS = -fopenmp

# GPU Threading
#CFLAGS = -fopenmp -target x86_64-pc-linux-gnu -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa -march=gfx90a


run: poisson2d.o poisson2d_serial.h
	$(CCOMP) $(CFLAGS) poisson2d.o -o run

poisson2d.o: poisson2d.c
	$(CCOMP) $(CFLAGS) -c poisson2d.c

.PHONY: clean cleanall

clean:
	rm -f run *.o


