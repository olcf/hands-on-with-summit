# OpenMP GPU Offload Basics

OpenMP is a directive-based programming model that allows you to write parallel code for multi-core, shared-memory processors - including GPUs. Here, we will focus on the GPU offloading capabilities of OpenMP. Instead of using a low-level programming method like CUDA, where the programmer is responsible for explicitly transferring data between the CPU and GPU and writing GPU kernels, with a directive-based model, the programmer simply adds "hints" within the code which tell the compiler where data transfers should happen and which code sections to offload to the GPU.

In this challenge, we will use a matrix-multiplication code to understand the very basics of programming GPUs with OpenMP offloading.

## Matrix Multiplication Code

In the following (serial) C code, we multiply two matrices of random numbers manually in a loop and then again using a call to the BLAS library call `cblas_dgemm`. The call to the math library serves two purposes; it gives us a correct answer to test our own results against, and (by timing it) it gives us a baseline time-to-solution to measure our own performance against.

```c
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

    // Stop timer for entire program
    stop_total = omp_get_wtime();
    total_elapsed_time = stop_total - start_total;

    // Print timing results
    printf("Elapsed time total (s)  : %16.14f\n", total_elapsed_time);
    printf("Elapsed time loop (s)   : %16.14f\n", loop_elapsed_time);
    printf("Elapsed time library (s): %16.14f\n", library_elapsed_time);

    return 0;
}
```

&nbsp;

Compiling and running this serial version of the code gives the following timing results:

```
Elapsed time total (s)  : 5818.85734658176079
Elapsed time loop (s)   : 5760.66759015014395
Elapsed time library (s): 51.50926541816443
```

Since it takes >90 minutes to perform the manual loop, you **should not run the serial version during this training**. However, the timing results above show that our matrix-multiply loop takes about 100x longer than the optimized matrix-multiply library call. 
 
&nbsp;

## Adding in OpenMP Offload directives

Now that we have a basic understanding of the serial code, let's try to speed up the manual matrix-multiply loop as a way to introduce the handful of OpenMP offload directives we'll cover in this challenge.

Before jumping into the code, it's important to first point out that the CPU and GPU are separate processors that have separate memories. The implications of this (in a very basic sense) are that the CPU acts as the "host" processor, going along executing the program with one or more CPU cores until it encounters a compute-intensive region of code (e.g., a matrix multiply) that can be offloaded to the GPU. At this point, the data needed to perform the calculations (e.g., the arrays/matrices for a matrix multiply) are transferred from the CPU to the GPU, the calculation (e.g., matrix multiply) is performed on the GPU, and the resulting data (i.e, the resulting matrix) are passed back to the CPU. Then, the CPU continues on executing the program until it reaches another compute-intensive region of the code, which it offloads to the GPU, and so on. 

So, if we (as programmers) want to offload work to a GPU, we typically need to 1) transfer data from the CPU to the GPU, 2) perform some calculations on the GPU, and 3) pass data back from the GPU to the CPU. Now let's see how to do this using OpenMP directives...


**`target` construct**

The first thing we'll look at is the `target` construct, which allows an executable region to be executed by a GPU - and can also be appended with data clauses that allow data transfers.

From the [OpenMP 4.5 Specification](https://www.openmp.org/wp-content/uploads/openmp-4.5.pdf),

```c
#pragma omp target [clause[ [,] clause] ... ] new-line
structured-block
``` 

So for our matrix-multiply loop, we would want something like this (where we'll need to change the placeholders for the clauses):

```c
    #pragma omp target [clause[ [,] clause] ... ]
    {

    for(int row=0; row<N; row++){
        for(int col=0; col<N; col++){
            for(int k=0; k<N; k++){

                int index = row * N + col;
                C[index] = C[index] + A[row * N + k] * B[k * N + col];

            }
        }
    }

    }
```

Now we can replace the clause placeholders with `map` clauses to tell the compiler how to transfer the necessary data to and from the GPU. From the [OpenMP 4.5 Specification](https://www.openmp.org/wp-content/uploads/openmp-4.5.pdf),

```c
map([ [map-type-modifier[,]] map-type : ] list)
```

Before adding in these map clauses, we should think about what data needs to be passed to and from the GPU to perform the matrix-multiply loop. In the loop, we can see that the C array is being updated, but the A and B arrays are only being read. So we will need to pass all 3 arrays *to the GPU*, since we'll need to know the values of A and B to perform the matrix multiplication, and since we'll need C to be intialized to zero. Then, after the matrix multiply is performed on the GPU, we'll need to pass the C array back to the CPU, since that is the ultimate result we're looking for. And since the A and B arrays are not changed (only read) during the matrix multiply, we do not need to pass those data back to the host (CPU).

&nbsp;

> NOTE: Because GPUs can perform calculations very quickly, the data transfers are often the bottleneck in a program's overall run time. So, whenever possible, we should avoid performing unnecessary data transfers.

&nbsp;

So the `map` clauses we'll need to append to the `target` construct are

```c
map(to:A[0:N*N],B[0:N*N]) map(tofrom:C[0:N*N])
```

where the "map-type" `to` tells the compiler to copy data to the GPU when the target region is encountered, and the "map-type" `tofrom` tells the compiler to copy data to the GPU when the target region is encountered and also to copy data back to the CPU when the end of the target region is encountered (e.g., the ending curly brace in the `#pragma omp target` region in this case). The range of the arrays in the `map` clauses should be read as `[<starting_index>:<number_of_indices_to_transfer>]`. Ok, now let's add these clauses into our code:


```c
    #pragma omp target map(to:A[:N*N],B[:N*N]) map(tofrom:C[:N*N])
    {
    
    for(int row=0; row<N; row++){
        for(int col=0; col<N; col++){
            for(int k=0; k<N; k++){

                int index = row * N + col;
                C[index] = C[index] + A[row * N + k] * B[k * N + col];

            }
        }
    }

    }
```

At this point, we have created a target region and told the compiler to offload this code to be run on the GPU along with the data that needs to be transferred. However, in its current form, the code would only be run on a single thread in a single thread block on the GPU (see [this NVIDIA blog post](https://developer.nvidia.com/blog/cuda-refresher-cuda-programming-model/) for an understanding of the grid-block-thread hierarchy).

To truly take advantage of the GPU, we'll need to add the `teams`, `distribute`, and `parallel for` constructs. The `teams` construct creates a league of thread teams (essentially a "grid" of "thread blocks"), the `distribute` construct distributes the iterations of a loop across the master threads of the thread teams (essentially across the "thread blocks"), and the parallel loop construct distributes the loop iterations (given to each team) across the threads of the teams. These constructs can be combined in a single directive, and places inside the `target` region as follows:

```c
    #pragma omp target map(to:A[:N*N],B[:N*N]) map(tofrom:C[:N*N])
    {

    #pragma omp teams distribute parallel for
    for(int row=0; row<N; row++){
        for(int col=0; col<N; col++){
            for(int k=0; k<N; k++){

                int index = row * N + col;
                C[index] = C[index] + A[row * N + k] * B[k * N + col];

            }
        }
    }

    }
```

&nbsp;

### Add OpenMP directives to the serial code, compile, and run
First, makes sure you're in the `OpenMP_Offload` challenge directory:

```bash
$ cd ~/hands-on-with-Frontier-/challenges/OpenMP_Offload
```

Then load the following modules:

```bash
$ module load PrgEnv-amd
$ module load craype-accel-amd-gfx90a
$ module load openblas
```

Now add the OpenMP directives above to the serial version of the code.

After you've done this, just issue the command `make`. 

Once you've successfully compiled the code, submit the job as follows:

```bash
$ sbatch submit.sbatch
```

You can monitor the progress of your job by issuing the command `sacct -u USERNAME`, where `USERNAME` should be replaced with your username. Once the job finishes, you can find the result in the output file, `mat_mul-JOBID.out`. If successful, the results should show the timing output of the job, which should look something similar to this:

```
Elapsed time total (s)  : 74.04765627099914
Elapsed time loop (s)   : 16.28946080500100
Elapsed time library (s): 51.09464657500212
```

From the results, we can see we've achieved a 350x speedup relative to our serial version of the matrix-multiply loop. 

However, in our current version of the code, we are only parallelizing the outermost `for` loop in the triply-nested loop, but the middle loop can also be parallelized (it's possible to parallelize the innermost loop too but we will not do so here). This can be accomplished in multiple ways, but for simplicity, we'll just append `collapse(2)` to the `parallel for` directive:

```c
    #pragma omp target map(to:A[:N*N],B[:N*N]) map(tofrom:C[:N*N])
    {

    #pragma omp teams distribute parallel for collapse(2)
    for(int row=0; row<N; row++){
        for(int col=0; col<N; col++){
            for(int k=0; k<N; k++){

                int index = row * N + col;
                C[index] = C[index] + A[row * N + k] * B[k * N + col];

            }
        }
    }

    }
```

This `collapse` clause tells the compiler to collapse the outer 2 loops and treat them as a single loop, which also causes the directive to be applied to the single "combined" loop. Now add this in to the code, recompile, and run the program. It should give you an additional ~3x speedup, for a total of >1000x speedup:

```
Elapsed time total (s)  : 62.66796130000148
Elapsed time loop (s)   : 5.26038305900147
Elapsed time library (s): 50.74522275100026
```

## Summary

So with minimal changes to the code, we were able to get a significant speedup of our matrix-multiply code. There is, of course, still some tuning that could be done (e.g. changing the number of teams, threads per team, etc.) for an even larger speedup, but the real purpose of this challenge was simply to give the participant a flavor of how OpenMP directives can be used to accelerate a code using a GPU.  

It should be noted that the directives covered in this short challenge really only scratch the surface of the GPU offload functionality in the OpenMP specification. If you'd like to learn more, please visit [https://www.openmp.org/specifications/](https://www.openmp.org/specifications/).

> NOTE: The OpenMP specification is "the collection of compiler directives, library routines, and environment variables" that define the OpenMP Application Program Interface (OpenMP API). A compiler can include "an implementation" of the OpenMP specification with either partial or full support.












