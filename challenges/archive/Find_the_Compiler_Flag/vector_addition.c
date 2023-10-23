#include <stdlib.h>
#include <stdio.h>

int main()
{
	// Number of elements in arrays
	unsigned int N = 1 << 20;

	// Bytes in arrays
	size_t bytes_in_array = N*sizeof(double);

	// Allocate memory for arrays on host
	double *A = (double*)malloc(bytes_in_array);
	double *B = (double*)malloc(bytes_in_array);
	double *C = (double*)malloc(bytes_in_array);

	// Initialize vector values
	for(int i=0; i<N; i++){
		A[i] = 1.0;
		B[i] = 2.0;
	}

	// Perform element-wise addition of vectors
	#pragma acc parallel loop
	for(int i=0; i<N; i++){
		C[i] = A[i] + B[i];
	}

	// Check for correctness
	for(int i=0; i<N; i++){
		if(C[i] != 3.0){
			printf("Error: Element C[%d] = %f instead of 3.0\n", i, C[i]);
			exit(0);
		}
	}

	printf("__SUCCESS__\n");
	
}



