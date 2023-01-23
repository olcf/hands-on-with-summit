#include <stdlib.h>
#include <stdio.h>

// Size of array
#define N 1048576

// Main program
int main()
{
	// Number of bytes to allocate for N doubles
	size_t bytes = N*sizeof(double);

	// Allocate memory for arrays A, B, and C
	double *A = (double*)malloc(bytes);
	double *B = (double*)malloc(bytes);
	double *C = (double*)malloc(bytes);

	// Fill arrays A and B
	for(int i=0; i<N; i++)
	{
		A[i] = 1.0;
		B[i] = 2.0;
	}

	// Add vectors (C = A + B)
	for(int i=0; i<N; i++)
	{
		C[i] = A[i] + B[i];
	}

	// Verify results
	for(int i=0; i<N; i++)
	{
		if(C[i] != 3.0)
		{ 
			printf("\nError: value of C[%d] = %f instead of 3.0\n\n", i, C[i]);
			exit(-1);
		}
	}	

	// Free memory
	free(A);
	free(B);
	free(C);

	printf("\n---------------------------\n");
	printf("__SUCCESS__\n");
	printf("---------------------------\n");

	return 0;
}
