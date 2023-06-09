#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cblas.h>
#include <omp.h>

int main(int argc, char *argv[]){

    // Declare variables for timing
    double start_total, stop_total, total_elapsed_time;
    double start, stop, loop_elapsed_time, library_elapsed_time;

    // Start timer for entire program
    start_total = omp_get_wtime();

    // Set value of N for N*N matrices
    int N = 8192;
    size_t buffer_size = N * N * sizeof(double);

    // Allocate memory for N*N matrices
    double *A     = (double*)malloc(buffer_size);
    double *B     = (double*)malloc(buffer_size);
    double *C     = (double*)malloc(buffer_size);
    double *ref_C = (double*)malloc(buffer_size);

    /* ---------------------------------------------------------------
    Set values of A, B, C to random values and C, ref_C to 0
    --------------------------------------------------------------- */
    for(int row=0; row<N; row++){
        for(int col=0; col<N; col++){

            int index = row * N + col;
            A[index]     = (double)rand()/(double)(RAND_MAX);
            B[index]     = (double)rand()/(double)(RAND_MAX);
            C[index]     = 0.0;
            ref_C[index] = 0.0;

        }
    } 

    /* ---------------------------------------------------------------
    Perform matrix multiply manually within a loop and measure time 
    --------------------------------------------------------------- */
    start = omp_get_wtime();

    for(int row=0; row<N; row++){
        for(int col=0; col<N; col++){
            for(int k=0; k<N; k++){

                int index = row * N + col;
                C[index] = C[index] + A[row * N + k] * B[k * N + col];

            }
        }
    }

    stop = omp_get_wtime();
    loop_elapsed_time = stop - start; 

    /* ---------------------------------------------------------------
    Perform matrix multiply using a math library and measure time
    --------------------------------------------------------------- */
    const double alpha = 1.0;
    const double beta  = 0.0;
    double tolerance   = 1.0e-12;
    
    start = omp_get_wtime();
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, N, N, N, alpha, A, N, B, N, beta, ref_C, N);
    stop = omp_get_wtime();
    library_elapsed_time = stop - start;

    /* ---------------------------------------------------------------
    Compare results of manual matrix multiply with library call
    --------------------------------------------------------------- */
    /* 
    for(int row=0; row<N; row++){
        for(int col=0; col<N; col++){

            int index = row * N + col;
            double difference = C[index] - ref_C[index];

            if( fabs(difference) > tolerance ){
                printf("Error: C[%d] - ref_C[%d] = %16.14f - %16.14f = %16.14f > tolerance (%16.14f)\n",
                       index, index, C[index], ref_C[index], fabs(difference), tolerance);
            }

        }
    }
    */
    // Causes program to error on every entry.



    // Stop timer for entire program
    stop_total = omp_get_wtime();
    total_elapsed_time = stop_total - start_total;

    // Print timing results
    printf("Elapsed time total (s)  : %16.14f\n", total_elapsed_time);
    printf("Elapsed time loop (s)   : %16.14f\n", loop_elapsed_time);
    printf("Elapsed time library (s): %16.14f\n", library_elapsed_time);

    return 0;
}
