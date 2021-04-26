CCOMP  = pgcc
CFLAGS =

INCLUDES  = -I$(OLCF_ESSL_ROOT)/include -I$(OLCF_CUDA_ROOT)/include
LIBRARIES = -L$(CUDA_DIR)/lib64 -L$(OLCF_ESSL_ROOT)/lib64 -lcublas -lessl -lcudart

cpu_gpu_dgemm: cpu_gpu_dgemm.o
	$(CCOMP) $(CFLAGS) $(LIBRARIES) cpu_gpu_dgemm.o -o cpu_gpu_dgemm

cpu_gpu_dgemm.o: cpu_gpu_dgemm.c
	$(CCOMP) $(CFLAGS) $(INCLUDES) -c cpu_gpu_dgemm.c

.PHONY: clean

clean:
	rm -f cpu_gpu_dgemm *.o

cleanall:
	rm -f cpu_gpu_dgemm *.o dgemm*
