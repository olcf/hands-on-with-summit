# OpenMP Basics

OpenMP is programming model that allows you to write parallel code for multi-core, shared-memory processors. Your laptop/desktop likely has a multi-core processor (e.g., 4-core). This is also true of each individual compute node on Frontier - each has 64 physical CPU cores that have access to 512 GB of DDR4 memory. By shared-memory, we simply mean that all the CPU cores have access to the same memory (DRAM). 

In this challenge, we will explore the very basics of the [OpenMP Application Program Interface (OpenMP API)](https://www.openmp.org/specifications/), which consists of a collection of compiler directives, library routines, and environment variables. In the examples below, we will insert compiler directives into the code to tell the compiler how the program should be executed in parallel, and we will also use a couple of API functions and environment variables along the way.

## Example Program 1: Hello, World!

To begin exploring OpenMP, we'll use a simple hello world program:

```c
#define _GNU_SOURCE

#include <stdio.h>
#include <sched.h>
#include <omp.h>

int main(int argc, char *argv[]){

  int num_threads;
  int thread_id;
  int virtual_core;

  #pragma omp parallel default(none) shared(num_threads) private(thread_id, virtual_core)
  {
    num_threads = omp_get_num_threads();
    thread_id = omp_get_thread_num();
    virtual_core  = sched_getcpu();

    printf("OpenMP thread %03d of %03d ran on virtual core %03d\n", thread_id, num_threads, virtual_core);
  }

  return 0;
}
```

Before running the code, let's break this down to understand what is being done. First, some ordinary integers are defined. But we're really interested in the line beginning with `#pragma omp` and the code following it within the curly braces. 

```c
#pragma omp parallel
```

When this directive is reached, a team of threads (each thread of execution typically runs on a separate CPU core) are spawned that will run the code within the (subsequent) curly braces in parallel. E.g., if 4 OpenMP threads are requested, then all 4 threads will run the code within the curly braces in parallel (concurrently). To clarify, all threads run the same code.

Inside the curly braces, we have

```c
num_threads = omp_get_num_threads();  // Find the number of OpenMP threads spawned
thread_id = omp_get_thread_num();     // Find a specific OpenMP thread's ID
virtual_core  = sched_getcpu();       // Find the virtual core the OpenMP thread ran on
```

`omp_get_num_threads()` and `omp_get_thread_num()` are two of the OpenMP API functions. They are used to determine the total number of OpenMP threads spawned inside the parallel region and a specific OpenMP thread's ID, respectively. In order to use these API calls, you need to include the `#include <omp.h>` header file. `sched_getcpu()` returns the ID of the virtual CPU core the OpenMP thread runs on - but this is **NOT** part of OpenMP.

> CLARIFICATION: Each Frontier node contains 64 physical CPU cores, and each physical CPU core has 2 virtual cores for a total of 128 total virtual cores per node. Everywhere else in the challenges, these virtual cores are referred to as hardware threads, but they will be referred to as virtual CPU cores here to avoid confusion with the use of the word threads in the context of OpenMP. 

Ok, great! But what about the rest of that OpenMP directive? Let's take a look at each one of the clauses individually:

```c
shared(num_threads)
```

The `shared` clause declares a variable to be "shared" among all threads in a parallel region - meaning that all threads access the same memory location for that variable. We can declare the variable `num_threads` (total number of OpenMP threads) as `shared` because each OpenMP thread will return the same value in the parallel region.

```c
private(thread_id, virtual_core)
```

The `private` clause declares a variable to be "private" to each thread in a parallel region - meaning that each thread has its own copy (and memory location) for that variable. We declare `thread_id` and `virtual_core` as `private` because each OpenMP thread will return its own unique value.

> CLARIFICATION: As mentioned in the beginning of this challenge, OpenMP is a "shared-memory" programming model, but that only means that threads in a parallel region need to *be able to* access the same memory. It doesn't mean that they *will* have access to the same memory locations - as shown with the `private` clause. Whether to declare something `shared` or `private` depends on the needs of the specific program.

```
default(none)
```

The `default` clause allows you to set the default value of privacy for variables. If set to `default(shared)`, then all variables will be assumed shared - but can be overridden by a separate `private` clause for specific variables. The default could also be set to e.g., `default(private)`. But many programmers prefer to set `default(none)` so that all variables must be manually declared as `shared`, `private`, or otherwise.

### Compile and Run the Code

Ok, now let's compile and run the code. First, make sure you're in the correct directory:

```
$ cd ~/hands-on-with-Frontier-/challenges/OpenMP_Basics/hello_world
```

Then, load the gcc compiler (if it's not already in your environment):

```
$ module load gcc
```

To compile, issue the command `make`. This uses the Makefile, which is a way of automating the compilation process. The commands that are actually run are shown as output:

```
$ make

gcc -fopenmp -c hello_world.c
gcc -fopenmp hello_world.o -o hello
```

The compiler flag `-fopenmp` tells the gcc compiler to act on the compiler directives in the code. If it is not included, the directives will be ignored.

Now we're ready to run! To do so, issue the command:

```
sbatch submit.sbatch
```

You can view the status of your job with the `sacct -u USERNAME` command. While you're waiting for the job to finish, take a look at the `submit.sbatch` script you used to submit your job. The environment variable `OMP_NUM_THREADS` can be used to set the number of OpenMP threads to be spawned in the parallel region. It's currently set to 4, but you can change it and re-run to see the results from different numbers of OpenMP threads.

Once your job is complete, you should have a file called `hello_test-JOBID.out`, where `JOBID` is the unique ID assigned to your job. This file will include the date, the output from the program, and some basic information about the job itself (below the dashed horizontal line). The program output should look something like this:

```
OpenMP thread 000 of 004 ran on virtual core 000
OpenMP thread 001 of 004 ran on virtual core 004
OpenMP thread 002 of 004 ran on virtual core 008
OpenMP thread 003 of 004 ran on virtual core 012
```

In the program, each OpenMP thread reports its thread ID and the virtual core it ran on. If you run the program several times, you'll see that the order each OpenMP thread reports its ID can change. This is because, although the threads are running in parallel, there is no guarantee that all threads will start or stop at exactly the same time.

## Example Program: Vector Addition

Next, we will use a vector addition program to understand how OpenMP can be used to speed up calculations - since this is really the goal of parallel programming. Vector addition is a simple but useful program to understand the basics of many parallel programming models because all the calulations can be performed "independently". If you are not familiar with vector addition, it is simply taking a vector, A:

```
A[0] = 1
A[1] = 1
...
``` 

and a vector, B:

```
B[0] = 2
B[1] = 2
...
```

and adding them element-wise, while capturing the results in a third vector, C:

```
C[0] = A[0] + B[0] = 1 + 2 = 3
C[1] = A[1] + B[1] = 1 + 2 = 3
...
```

As you can see, calculating element 0 (involving only elements `A[0], B[0], and C[0]`) does not affect the calculation of element 1 (involving only elements `A[1], B[1], and C[1]`), or the calculation of any other elements. So, the calculations are all "independent". 

> NOTE: Now imagine if calculating element 1 was dependent on the results of element 0 (e.g., C[1] = C[0] + A[1]).  In this case, you would have a "race condition", where C[1] could have a different value depending on whether or not C[0] was calculated before or after C[1].

Ok, let's take a look at a serial vector addition code:

```c
#include <stdlib.h>
#include <stdio.h>

int main()
{
    // Number of elements in arrays
    const int N = 1e8;

    // Bytes in arrays
    size_t bytes_in_array = N*sizeof(double);

    // Allocate memory for arrays
    double *A = (double*)malloc(bytes_in_array);
    double *B = (double*)malloc(bytes_in_array);
    double *C = (double*)malloc(bytes_in_array);

    // Initialize vector values
    for(int i=0; i<N; i++){
        A[i] = 1.0;
        B[i] = 2.0;
    }

    // Perform element-wise addition of vectors
    for(int i=0; i<N; i++){
        C[i] = A[i] + B[i];
    }

    // Check for correctness
    for(int i=0; i<N; i++){
        if(C[i] != 3.0){
            printf("Error: Element C[%d] = %f instead of 3.0\n", i, C[i]);
            exit(1);
        }
    }

    printf("__SUCCESS__\n");

    free(A);
    free(B);
    free(C);

    return 0;
}
```

This code should be simple enough to understand from the in-line comments (allocate memory for vectors/arrays, intialize values, perform element-wise vector addition, check results). 

So how would we go about parallelizing the "element-wise vector addition" loop with OpenMP? The first thing you might think of is to add a `#pragma omp parallel` region around the loop, with appropriate privacy clauses (as we did in the hello world example above):

```c
    // Perform element-wise addition of vectors
    #pragma omp parallel default(none) shared(A, B, C)
    {
      for(int i=0; i<N; i++){
        C[i] = A[i] + B[i];
      }
    }
```

However, thinking back to the hello world example, that would cause each thread to execute the code inside the curly braces; i.e., each thread would iterate over the full loop, performing the additions. This would still give the correct results, but if each thread still has to execute the full loop, we're not going to get a speedup. 

What we need instead is a way to divide the loop iterations among the available threads, so that each thread is only calculating the additions for a subset of the iterations. Then, if all threads are performing their part of the loop in parallel (at the same time), we would obtain a speedup.

So, how do we accomplish this? Well, we could manually create loop bounds (e.g., start and stop elements for each thread - based on its unique thread ID) so that each thread only works on a unique portion of array elements, but it's easier to just use a "work sharing construct". The most common one is the `for` directive:

```c
    // Perform element-wise addition of vectors
    #pragma omp parallel default(none) shared(A, B, C)
    {
      #pragma omp for
      for(int i=0; i<N; i++){
        C[i] = A[i] + B[i];
      }
    }  
```

The `#pragma omp for` instructs the compiler to distribute the loop iterations (of the loop following the `#pragma`) among the spawned OpenMP threads (i.e., the threads spawned within the `omp parallel` region surrounding the `omp for`). These two separate directives can actually be written as a single compound directive as follows:

```c
#pragma omp parallel for default(none) shared(A, B, C)
```

Similar to the hello world example, `default(none)` just says that the variables within the parallel region do not have a default privacy set. `shared(A, B, C)` says that vectors `A`, `B`, and `C` will be shared among all thread (i.e., all threads will access the same memory locations for the 3 vectors). This might seem incorrect at first, but, although all threads have access to the same memory for the vectors, each thread will only be updating its portion of the elements (thanks to the `for` work sharing construct).

> NOTE: Local variables (i.e., those defined within a parallel region) are always private by default. So in the for loop above, the `i` declared in the loop is private to each thread.

The version of the code included in this directory already has the directives above added in. One other change that was made was to include a timer: `omp_get_wtime()`. It is called before and after the parallel region, and the difference returns the wall clock time in seconds.

### Compile and Run the Code

Ok, now let's compile and run the code. First, make sure you're in the correct directory:

```
$ cd ~/hands-on-with-Frontier-/challenges/OpenMP_Basics/vector_addition
```

Then, load the gcc compiler (if it's not already in your environment):

```
$ module load gcc
```

To compile, issue the command `make`:

```
$ make

gcc -fopenmp -c vector_addition.c
gcc -fopenmp vector_addition.o -o vec_add
```

Now, we're ready to run! To do so, issue the command:

```
sbatch submit.sbatch
```

You can view the status of your job with the `sacct -u USERNAME` command. While you're waiting for the job to finish, take a look at the `submit.sbatch` script you used to submit your job. The environment variable `OMP_NUM_THREADS` can be used to set the number of OpenMP threads that are spawned in the parallel region. It's originally set to 4, but you can change it and re-run to see the results from different numbers of OpenMP threads.

Once your job is complete, you should have a file called `vec_add-JOBID.out`, where `JOBID` is the unique ID assigned to your job. This file will include the date, the output from the program, and some basic information about the job itself (below the dashed horizontal line). The program output should look something like this:

```
Number of OpenMP threads: 004
Elapsed Time (s)        : 0.288272
```

As you can see, the output is simply the total number of OpenMP threads spawned in the parallel region and the time taken to complete the vector addition loop. As mentioned above, you are encouraged to re-run the code with different numbers of OpenMP threads to see how it affects the total run time. 

## Summary

We really only scratched the surface of the OpenMP programming model in this challenge, but hopefully it gives a flavor of how OpenMP can be used to achieve improved performace from directive-based parallel programming. To learn more about OpenMP, you might start with the following links:

* [OpenMP Website](https://www.openmp.org/)
* [Lawrence Livermore National Laboratory (LLNL) OpenMP Tutorial](https://computing.llnl.gov/tutorials/openMP/)
* [National Energy Research Scientific Computing Center (NERSC) OpenMP Resources](https://docs.nersc.gov/development/programming-models/openmp/openmp-resources/)
