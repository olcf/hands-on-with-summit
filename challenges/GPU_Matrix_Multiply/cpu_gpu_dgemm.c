/*------------------------------------------------------------------------------------------------
This program will fill 2 NxN matrices with random numbers, compute a matrix multiply on the CPU 
and then on the GPU, compare the values for correctness, and print _SUCCESS_ (if successful).

Written by Tom Papatheodore
------------------------------------------------------------------------------------------------*/

#include <stdio.h>
#include <essl.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

// Macro for checking errors in CUDA API calls
#define cudaErrorCheck(call)                                                              \
do{                                                                                       \
    cudaError_t cuErr = call;                                                             \
    if(cudaSuccess != cuErr){                                                             \
      printf("CUDA Error - %s:%d: '%s'\n", __FILE__, __LINE__, cudaGetErrorString(cuErr));\
      exit(0);                                                                            \
    }                                                                                     \
}while(0)

#define N 512

int main(int argc, char *argv[])
{

	/* Set device to GPU 0 -------------------------------------------------------------*/
	cudaErrorCheck( cudaSetDevice(0) );


	/* Allocate memory for A, B, C on CPU ----------------------------------------------*/
	double *A = (double*)malloc(N*N*sizeof(double));
	double *B = (double*)malloc(N*N*sizeof(double));
	double *C = (double*)malloc(N*N*sizeof(double));


	/* Set Values for A, B, C on CPU ---------------------------------------------------*/
	double max_value = 10.0; /* Max size of random double */

	for(int i=0; i<N; i++){
		for(int j=0; j<N; j++){
			A[i*N + j] = (double)rand()/(double)(RAND_MAX/max_value);
			B[i*N + j] = (double)rand()/(double)(RAND_MAX/max_value);
			C[i*N + j] = 0.0;
		}
	}


	/* Allocate memory for d_A, d_B, d_C on GPU ----------------------------------------*/
	double *d_A, *d_B, *d_C;
	cudaErrorCheck( cudaMalloc(&d_A, N*N*sizeof(double)) );
	cudaErrorCheck( cudaMalloc(&d_B, N*N*sizeof(double)) );
	cudaErrorCheck( cudaMalloc(&d_C, N*N*sizeof(double)) );


	/* Copy host arrays (A,B,C) to device arrays (d_A,d_B,d_C) -------------------------*/
	cudaErrorCheck( cudaMemcpy(d_A, A, N*N*sizeof(double), cudaMemcpyHostToDevice) );
	cudaErrorCheck( cudaMemcpy(d_B, B, N*N*sizeof(double), cudaMemcpyHostToDevice) );
	cudaErrorCheck( cudaMemcpy(d_C, C, N*N*sizeof(double), cudaMemcpyHostToDevice) );


	/* Perform Matrix Multiply on CPU --------------------------------------------------*/
	const double alpha = 1.0;
	const double beta = 0.0;

	dgemm("n", "n", N, N, N, alpha, A, N, B, N, beta, C, N);


	/* Perform Matrix Multiply on GPU --------------------------------------------------*/
	cublasHandle_t handle;
	cublasCreate(&handle);


	/************************************************************/
	/* TODO: Look up the cublasDgemm routine and add it here to */
	/*       perform a matrix multiply on the GPU               */
	/*                                                          */
	/* NOTE: This will be similar to the CPU dgemm above but    */
	/*       will use d_A, d_B, and d_C instead                 */ 
	/************************************************************/


	/* Copy values of d_C (computed on GPU) into host array C_fromGPU ----------------- */
	double *C_fromGPU = (double*)malloc(N*N*sizeof(double));
	cudaErrorCheck( cudaMemcpy(C_fromGPU, d_C, N*N*sizeof(double), cudaMemcpyDeviceToHost) );


	/* Check if CPU and GPU give same results ----------------------------------------- */
	double tolerance = 1.0e-13;
	for(int i=0; i<N; i++){
		for(int j=0; j<N; j++){
			if(fabs((C[i*N + j] - C_fromGPU[i*N + j])/C[i*N + j]) > tolerance){
				printf("Element C[%d][%d] (%f) and C_fromGPU[%d][%d] (%f) do not match!\n", i, j, C[i*N + j], i, j, C_fromGPU[i*N + j]);
				return EXIT_FAILURE;
			}
		}
	}


	/* Clean up and output --------------------------------------------------------------*/

	cublasDestroy(handle);

	cudaErrorCheck( cudaFree(d_A) );
	cudaErrorCheck( cudaFree(d_B) );
	cudaErrorCheck( cudaFree(d_C) );

	free(A);
	free(B);
	free(C);
	free(C_fromGPU);

	printf("__SUCCESS__\n");

	return 0;
}
