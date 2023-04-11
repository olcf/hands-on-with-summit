COMP  = cc
FLAGS = -fopenmp -target x86_64-pc-linux-gnu -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa -march=gfx90a

INCLUDES  = -I${OLCF_OPENBLAS_ROOT}/include
LIBRARIES = -L${OLCF_OPENBLAS_ROOT}/lib -lopenblas

matrix_multiply: matrix_multiply.o
	${COMP} ${FLAGS} ${LIBRARIES} matrix_multiply.o -o matrix_multiply

matrix_multiply.o: matrix_multiply.c
	${COMP} ${FLAGS} ${INCLUDES} -c matrix_multiply.c

.PHONY: clean cleanall

clean:
	rm -f matrix_multiply *.o

cleanall:
	rm -f matrix_multiply *.o
