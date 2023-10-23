/* Copyright (c) 2016, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <math.h>
#include <string.h>
#include <stdio.h>
#include <sys/time.h>
#include <stdlib.h>

#ifdef _OPENACC
#include <openacc.h>
#endif /* _OPENACC */

#ifdef _OPENMP
#include <omp.h>
#endif /* _OPENMP */

#define max(x, y) (((x) > (y)) ? (x) : (y))
#define min(x, y) (((x) < (y)) ? (x) : (y))

#define NY 4096
#define NX 4096

double A[NY][NX];
double Anew[NY][NX];
double rhs[NY][NX];

double A_ref[NY][NX];
double Anew_ref[NY][NX];

#include "poisson2d_serial.h"
void poisson2d_serial(int , double);

int main(int argc, char** argv)
{
	// Set to 1 to run serial test, otherwise 0
	int serial_test = 0;

    int iter_max = 1000;
    const double tol = 1.0e-5;

	struct timeval start_time, stop_time, elapsed_time_serial, elapsed_time_parallel;

    int num_threads = 1;
    int thread_num  = 0;

    double global_error = 0.0;

    #pragma omp parallel default(shared) firstprivate(num_threads, thread_num)
    {

    /* ---------------------------------------------
        Set up OpenMP and OpenACC and
		initialize arrays
    --------------------------------------------- */

#ifdef _OPENMP
    num_threads = omp_get_num_threads();
    thread_num  = omp_get_thread_num();
#endif /* _OPENMP */

#ifdef _OPENACC
    int num_devices = acc_get_num_devices(acc_device_nvidia);
    int device_num  = thread_num % num_devices;
    acc_set_device_num(device_num, acc_device_nvidia);
#endif /* _OPENACC */

    #pragma omp master
    {
    // Set rhs
    for (int iy = 1; iy < NY-1; iy++)
    {
        for( int ix = 1; ix < NX-1; ix++ )
        {
            const double x = -1.0 + (2.0*ix/(NX-1));
            const double y = -1.0 + (2.0*iy/(NY-1));
            rhs[iy][ix] = exp(-10.0*(x*x + y*y));
        }
    }
 	} /* pragma omp master */

	// Set A and A_ref
	#pragma acc kernels
    for(int iy = 0; iy < NY; iy++)
    {
        for(int ix = 0; ix < NX; ix++)
        {
            A_ref[iy][ix] = 0.0;
            A[iy][ix]    = 0.0;
        }
    }

    /* ---------------------------------------------
        Single-GPU execution
    --------------------------------------------- */
    #pragma omp master
    {
	if(serial_test == 1)
	{
		printf("Jacobi relaxation Calculation: %d x %d mesh\n", NY, NX);
		printf("Single-GPU Execution...\n");

		// Start single-GPU timer
		gettimeofday(&start_time, NULL);

		// Run single-GPU version
		poisson2d_serial(iter_max, tol);

		// Stop single-GPU timer
		gettimeofday(&stop_time, NULL);
		timersub(&stop_time, &start_time, &elapsed_time_serial);
	}
	} /* pragma omp master */

	#pragma omp barrier

    /* ---------------------------------------------
        Parallel Execution
    --------------------------------------------- */
	#pragma omp master
	{
	printf("Parallel Execution...\n"); 

	// Start parallel timer
	gettimeofday(&start_time, NULL);
	}

    int iter  = 0;
    double error = 1.0;
  
    int ix_start = 1;
    int ix_end   = NX;

    // Use ceil function in case num_threads does not divide evenly into NY
    int chunk_size = ceil((1.0*NY)/num_threads);

	// For each thread, these values are set so the loops below can iterate
	// from iy_start to iy_end-1, which include only the inner region of the 
	// domain that need to be calculated.
	//
	// They are also used to set the ranges of data that each thread sends
	// to its GPU (including the halo region).
    int iy_start = thread_num * chunk_size;
    int iy_end   = iy_start + chunk_size;

    // Only process inner region - not boundaries
    iy_start = max(iy_start, 1);
    iy_end   = min(iy_end, NY-1);
 
	#pragma acc data copy(A[(iy_start-1):(iy_end-iy_start)+2][0:NX]) copyin(rhs[iy_start:(iy_end-iy_start)][0:NX]) create(Anew[iy_start:(iy_end-iy_start)][0:NX])
	{
	// Main iteration loop
    while ( error > tol && iter < iter_max )
    {

        error = 0.0;
        #pragma omp single
        {
        global_error = 0.0;
        }
        #pragma omp barrier

		#pragma acc kernels
        for (int iy = iy_start; iy < iy_end; iy++)
        {
            for( int ix = ix_start; ix < ix_end; ix++ )
            {
                Anew[iy][ix] = -0.25 * (rhs[iy][ix] - ( A[iy][ix+1] + A[iy][ix-1]
                                                       + A[iy-1][ix] + A[iy+1][ix] ));
                error = fmax( error, fabs(Anew[iy][ix]-A[iy][ix]));
            }
        }

        #pragma omp critical
        {
        global_error = max(global_error, error);
        }       

        #pragma omp barrier
        error = global_error;
 
		#pragma acc kernels
        for (int iy = iy_start; iy < iy_end; iy++)
        {
            for( int ix = ix_start; ix < ix_end; ix++ )
            {
                A[iy][ix] = Anew[iy][ix];
            }
        }
        
        // Begin periodic boundary conditions update
        #pragma acc update self(A[iy_start:1][0:NX], A[(iy_end-1):1][0:NX])

        #pragma omp barrier
        if(0 == (iy_start-1))
        {
            for( int ix = 1; ix < NX-1; ix++ )
            {
                A[0][ix]      = A[(NY-2)][ix];
            }
        }

        if((NY-1) == (iy_end))
        {
            for( int ix = 1; ix < NX-1; ix++ )
            {
                A[(NY-1)][ix] = A[1][ix];
            }
        }

        #pragma acc update device(A[(iy_start-1):1][0:NX], A[iy_end:1][0:NX])

        #pragma acc kernels
        for (int iy = iy_start; iy < iy_end; iy++)
        {
                A[iy][0]      = A[iy][(NX-2)];
                A[iy][(NX-1)] = A[iy][1];
        }

        #pragma omp master
        {
        if((iter % 100) == 0) printf("%5d, %0.6f\n", iter, error);
        }

        iter++;
    }
	} /* #pragma acc data */

	#pragma omp barrier

    /* ---------------------------------------------
        Stop parallel timer and print results
    --------------------------------------------- */
	#pragma omp master
	{
	// Stop parallel timer
	gettimeofday(&stop_time, NULL);
	timersub(&stop_time, &start_time, &elapsed_time_parallel);

	double runtime_parallel = elapsed_time_parallel.tv_sec+elapsed_time_parallel.tv_usec/1000000.0;

	if(serial_test == 1)
	{	
		// Compare A and A_ref 
		for(int iy = 0; iy < NY; iy++)
		{
			for(int ix = 0; ix < NX; ix++)
			{
				if( abs(A_ref[iy][ix] - A[iy][ix]) > tol )
				{
					printf("A_ref[%d][%d] - A[%d][%d] = %f\n", iy, ix, iy, ix, A_ref[iy][ix] - A[iy][ix]);
					printf("Exiting...\n");
					exit(1);
				}
			}
		}

		double runtime_serial   = elapsed_time_serial.tv_sec+elapsed_time_serial.tv_usec/1000000.0;
		printf("Elapsed Time (s) - Serial: %8.4f, Parallel: %8.4f, Speedup: %8.4f\n", runtime_serial, runtime_parallel, runtime_serial/runtime_parallel);
	}
	else
	{
		printf("Elapsed Time (s) - Parallel: %8.4f\n", runtime_parallel);
	}

	} /* pragma omp master */

    } /* #pragma omp parallel */

    return 0;
}
