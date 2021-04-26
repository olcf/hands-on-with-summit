COMP  = xlc_r
FLAGS = -qsmp=omp

INCLUDES  = -I${OLCF_ESSL_ROOT}/include
LIBRARIES = -L${OLCF_ESSL_ROOT}/lib64 -lessl

matrix_multiply: matrix_multiply.o
	${COMP} ${FLAGS} ${LIBRARIES} matrix_multiply.o -o matrix_multiply

matrix_multiply.o: matrix_multiply.c
	${COMP} ${FLAGS} ${INCLUDES} -c matrix_multiply.c

.PHONY: clean cleanall

clean:
	rm -f matrix_multiply *.o

cleanall:
	rm -f matrix_multiply *.o
