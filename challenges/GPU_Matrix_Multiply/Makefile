COMP = CC
CFLAGS = -std=c++11 -D__HIP_ROCclr__ -D__HIP_ARCH_GFX90A__=1 --rocm-path=${ROCM_PATH} --offload-arch=gfx90a -x hip
LFLAGS = --rocm-path=${ROCM_PATH}

INCLUDES  = -I${OLCF_OPENBLAS_ROOT}/include -I${OLCF_ROCM_ROOT}/include
LIBRARIES = -L${OLCF_OPENBLAS_ROOT}/lib -lopenblas  -L${ROCM_PATH}/lib -lhipblas -lamdhip64 

cpu_gpu_dgemm: cpu_gpu_dgemm.o
	${COMP} ${LFLAGS} ${LIBRARIES} cpu_gpu_dgemm.o -o cpu_gpu_dgemm

cpu_gpu_dgemm.o: cpu_gpu_dgemm.cpp
	${COMP} ${CFLAGS} ${INCLUDES} -c cpu_gpu_dgemm.cpp

.PHONY: clean

clean:
	rm -f cpu_gpu_dgemm *.o