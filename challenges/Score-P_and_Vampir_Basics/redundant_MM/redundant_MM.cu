/*------------------------------------------------------------------------------------------------
redundant_MM

For each MPI rank, this program does the following:
  * Fill 2 NxN matrices with random numbers
  * Compute a matrix multiply on the CPU
	* Compute a matrix multiply on the GPU (loop_count times)
  * Compare the CPU and GPU results for consistency
  * Output the total runtime and time spent computing on the GPUs for each rank (and max)
    as well as the hardware thread and GPU used on a specific node

USAGE:

Two command line arguments must be supplied:
	N (matrix size)
	loop_count (number of times cublasDgemm is called)

For example,

	$ jsrun -n6 -c1 -g1 -a1 -r3 ./redundant_MM 2048 1000 | sort
	(N = 2048) Max Total Time: 6.879220 Max GPU Time: 2.816899
	Rank 000, HWThread 002, GPU 0, Node h41n09 - Total Time: 6.855115 GPU Time: 2.804994
	Rank 001, HWThread 004, GPU 1, Node h41n09 - Total Time: 6.816647 GPU Time: 2.814934
	Rank 002, HWThread 008, GPU 2, Node h41n09 - Total Time: 6.879220 GPU Time: 2.816899
	Rank 003, HWThread 000, GPU 0, Node h41n10 - Total Time: 5.862273 GPU Time: 2.814339
	Rank 004, HWThread 005, GPU 1, Node h41n10 - Total Time: 5.798143 GPU Time: 2.765094
	Rank 005, HWThread 010, GPU 2, Node h41n10 - Total Time: 5.746687 GPU Time: 2.785626

Written by Tom Papatheodore
------------------------------------------------------------------------------------------------*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sched.h>
#include <mpi.h>
#include <essl.h>
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

int main(int argc, char *argv[])
{

	/* -------------------------------------------------------------------------------------------
		MPI Initialization 
	--------------------------------------------------------------------------------------------*/
	MPI_Init(&argc, &argv);

	int size;
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	char name[MPI_MAX_PROCESSOR_NAME];
	int resultlength;
	MPI_Get_processor_name(name, &resultlength);
	
	const char* nl_rank = getenv("OMPI_COMM_WORLD_LOCAL_RANK");
	int node_local_rank = atoi(nl_rank);

	/* -------------------------------------------------------------------------------------------
		Other Initialization 
	--------------------------------------------------------------------------------------------*/

	// Start Total Runtime Timer
	double start_time, end_time, elapsed_time;
	start_time = MPI_Wtime();

	// Matrix size
	int N;

	// Number of times cublasDgemm is called
	int loop_count;

	// Check for proper command line arguments
	if(argc != 3){
		printf("Must supply two arguments: N (matrix size) and loop_count (number of cublasDgemm calls). Exiting...\n");
		exit(0);
	}
	else{
		for(int i=0; i<strlen(argv[1]); i++){
			if(!isdigit(argv[1][i])){
				printf("1st argument must be a positive integer! Exiting...\n");
				exit(0);
			}
		}
		N = atoi(argv[1]);		

		for(int i=0; i<strlen(argv[2]); i++){
			if(!isdigit(argv[2][i])){
				printf("2nd argument must be a positive integer! Exiting...\n");
				exit(0);
			}
		}
		loop_count = atoi(argv[2]);
	}

	// Find hardware thread being used by each MPI rank
	int hwthread = sched_getcpu();

	// Find how many GPUs CUDA runtime says are available
	int num_devices = 0;
	cudaErrorCheck( cudaGetDeviceCount(&num_devices) );

	// Map MPI ranks to GPUs according to node-local MPI rank (round-robin)
	int gpu_id = node_local_rank % num_devices;
	cudaErrorCheck( cudaSetDevice(gpu_id) );

	/* -------------------------------------------------------------------------------------------
		Allocate memory for arrays on CPU and GPU
	--------------------------------------------------------------------------------------------*/

	// Allocate memory for A, B, C on CPU
	double *A = (double*)malloc(N*N*sizeof(double));
	double *B = (double*)malloc(N*N*sizeof(double));
	double *C = (double*)malloc(N*N*sizeof(double));

    // Allocate memory for d_A, d_B, d_C on GPU
    double *d_A, *d_B, *d_C;
    cudaErrorCheck( cudaMalloc(&d_A, N*N*sizeof(double)) );
    cudaErrorCheck( cudaMalloc(&d_B, N*N*sizeof(double)) );
    cudaErrorCheck( cudaMalloc(&d_C, N*N*sizeof(double)) );

    /* -------------------------------------------------------------------------------------------
        Fill arrays on CPU
    --------------------------------------------------------------------------------------------*/

	// Max size of random double
	double max_value = 10.0;

	// Set A, B, and C
	for(int i=0; i<N; i++){
		for(int j=0; j<N; j++){
			A[i*N + j] = (double)rand()/(double)(RAND_MAX/max_value);
			B[i*N + j] = (double)rand()/(double)(RAND_MAX/max_value);
			C[i*N + j] = 0.0;
		}
	}

    /* -------------------------------------------------------------------------------------------
        Transfer data from CPU to GPU
    --------------------------------------------------------------------------------------------*/

	// Copy host arrays (A,B,C) to device arrays (d_A,d_B,d_C)
	cudaErrorCheck( cudaMemcpy(d_A, A, N*N*sizeof(double), cudaMemcpyHostToDevice) );
	cudaErrorCheck( cudaMemcpy(d_B, B, N*N*sizeof(double), cudaMemcpyHostToDevice) );
	cudaErrorCheck( cudaMemcpy(d_C, C, N*N*sizeof(double), cudaMemcpyHostToDevice) );	

	/* -------------------------------------------------------------------------------------------
		Perform DGEMM on CPU
	--------------------------------------------------------------------------------------------*/

	const double alpha = 1.0;
	const double beta = 0.0;

	// Perform Matrix Multiply on CPU
	dgemm("n", "n", N, N, N, alpha, A, N, B, N, beta, C, N);

    /* -------------------------------------------------------------------------------------------
        Perform DGEMM on GPU (loop_count times) and time GPU execution
    --------------------------------------------------------------------------------------------*/

	cublasHandle_t handle;
	cublasCreate(&handle);

	cudaEvent_t start_gpu, end_gpu;
	cudaEventCreate(&start_gpu);
	cudaEventCreate(&end_gpu);

	// Start GPU timer
	cudaEventRecord(start_gpu);

	for(int i=0; i<loop_count; i++){
		// Perform Matrix Multiply on GPU
		cublasStatus_t status = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, d_A, N, d_B, N, &beta, d_C, N);
		if (status != CUBLAS_STATUS_SUCCESS){
			printf("cublasDgemm failed with code %d\n", status);
			return EXIT_FAILURE;
		}
	}

	// Stop GPU timer
	cudaEventRecord(end_gpu);
	cudaEventSynchronize(end_gpu);
	float milliseconds = 0.0;
	float seconds;

	cudaEventElapsedTime(&milliseconds, start_gpu, end_gpu);
	seconds = milliseconds / 1000;

	cublasDestroy(handle);

    /* -------------------------------------------------------------------------------------------
        Transfer results from GPU DGEMM to CPU
    --------------------------------------------------------------------------------------------*/

	// Copy values of d_C computed on GPU into host array C_fromGPU	
	double *C_fromGPU = (double*)malloc(N*N*sizeof(double));	
	cudaErrorCheck( cudaMemcpy(C_fromGPU, d_C, N*N*sizeof(double), cudaMemcpyDeviceToHost) );

    /* -------------------------------------------------------------------------------------------
        Check for consistency between the CPU and GPU results
    --------------------------------------------------------------------------------------------*/

	// Check if CPU and GPU give same results
	double tolerance = 1.0e-13;
	for(int i=0; i<N; i++){
		for(int j=0; j<N; j++){
			if(fabs((C[i*N + j] - C_fromGPU[i*N + j])/C[i*N + j]) > tolerance){
				printf("Element C[%d][%d] (%f) and C_fromGPU[%d][%d] (%f) do not match!\n", i, j, C[i*N + j], i, j, C_fromGPU[i*N + j]);
				return EXIT_FAILURE;
			}
		}
	}

	/* -------------------------------------------------------------------------------------------
		Clean up memory and stop timer
	--------------------------------------------------------------------------------------------*/

	// Free GPU memory
	cudaErrorCheck( cudaFree(d_A) );
	cudaErrorCheck( cudaFree(d_B) );
	cudaErrorCheck( cudaFree(d_C) );


	// Free CPU memory
	free(A);
	free(B);
	free(C);
	free(C_fromGPU);

	// End Total Runtime Timer
	end_time = MPI_Wtime();
	elapsed_time = end_time - start_time;

	/* -------------------------------------------------------------------------------------------
		MPI Reductions to find the maximum total runtime and maximum time spent computing on GPUs.
		(These are used as proxies for total runtime and total time spent computing on GPUs)
	--------------------------------------------------------------------------------------------*/

	double total_time_max;
	MPI_Reduce(&elapsed_time, &total_time_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

	float gpu_time_max;
	MPI_Reduce(&seconds, &gpu_time_max, 1, MPI_FLOAT, MPI_MAX, 0, MPI_COMM_WORLD);

	/* -------------------------------------------------------------------------------------------
		Output and finalize
	--------------------------------------------------------------------------------------------*/

	// MPI rank 0 will output the maximum total runtime and maximum time spent computing on GPUs
	if(rank == 0){
		printf("(N = %d) Max Total Time: %f Max GPU Time: %f\n", N, total_time_max, gpu_time_max);
	}

	// Each MPI rank will output its total runtime and time spent computing on GPUs
	printf("Rank %03d, HWThread %03d, GPU %d, Node %s - Total Time: %f GPU Time: %f\n", rank, hwthread, gpu_id, name, elapsed_time, seconds); 

	MPI_Finalize();

	return 0;
}
