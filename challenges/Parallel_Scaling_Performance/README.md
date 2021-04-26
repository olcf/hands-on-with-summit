# Parallel Scaling Performance

In high-performance computing, "scaling" refers to how the performance of a parallel application changes as the number of processing element increases. The term "strong scaling" is used when the problem size stays the same as the number of processing elements increases, and the term "weak scaling" is used when the problem size increases as the number of processing elements increases (so the amount of work done by a single processing element stays roughly the same). In this challenge, you will plot the strong scaling results of an application.

> NOTE: You will not need to edit any code. You will simply need to compile and run the code with different numbers of OpenMP threads and GPUs.

## Set Up Your Programming Environment and Compile

First, you'll need to make sure you're in the `Parallel_Scaling_Performance/` directory:

```
$ cd ~/CSE21_HandsOn_with_Summit/challenges/Parallel_Scaling_Performance
```

Next, load the PGI compiler and CUDA Toolkit:

```
$ module load pgi cuda
```

Then compile the program:

```
$ make
```

## Run the Code with Increasing Numbers of OpenMP Threads

The program you will be running is a Jacobi iteration solver for a Poisson equation, but the exact details of the code will not be explained. It is sufficient to know that the program performs calculations on all the elements of a 4096 X 4096 matrix. Within the code, the OpenMP programming model is used to split up the work (the calculations on all the matrix elements) among multiple "threads" that run on different CPU cores. So each thread can perform the calculations on its portion of the matrix elements at the same time as other threads perform the calculations on their portion of the matrix elements. By splitting up the work among more threads, and performing the calculations at the same time, the code can be run faster.

Submit the `submit_01.lsf`, `submit_02.lsf`, `submit_04.lsf`, `submit_08.lsf`, `submit_16.lsf`, `submit_32.lsf`, and `submit_42.lsf` scripts to run the program with 1, 2, 4, 8, 16, 32, and 42 OpenMP threads:

```
$ bsub submit_01.lsf
$ bsub submit_02.lsf
...
```

> NOTE: There are 42 physical cores on each Summit node (21 per Power9 processor), which is why the 42-thread version is included (but does not fit the pattern of doubling the threads).

You can check whether your jobs are running, eligible (waiting), or finished using the following command:

```
$ jobstat -u USERNAME
```
where `USERNAME` should be replaced with your username.

## Plot the Speedup vs Number of OpenMP Threads

Once the jobs have finished, the runtime of the 7 jobs can be found in the output files labeled `01_omp.JOBID`, `02_omp.JOBID`, `04_omp.JOBID`, `08_omp.JOBID`, `16_omp.JOBID`, `32_omp.JOBID`, and `42_omp.JOBID`, where `JOBID` is the unique ID assigned to each job.

Now calculate the "speedup" of each run relative to the single OpenMP thread case. The speedup for each run can be calculated by dividing the runtime of the single-OpenMP-thread job by the runtime of the multiple-OpenMP-thread job. For example,

```
(speedup of 4 OpenMP thread job) = (runtime of 1 OpenMP thread job) / (runtime of 4 OpenMP thread job) 
```

> NOTE: The speedup for the single-OpenMP-thread job will be 1 (since you're dividing its runtime by its own runtime).

Now use whatever method you are familar with (Python, Excel, etc.) to plot the speedup versus number of OpenMP threads for the 7 jobs (on the same plot). 

## Compile the Code with GPU-Support

The code we've been running also includes OpenACC directives, but we haven't compiled the code to recognize them. Now let's include this GPU support. To do so, comment out (add `#`) line 3 and uncomment line 2 in the `Makefile` so it looks as follows:

```
CCOMP = pgcc
CFLAGS = -acc -Minfo=acc -ta=tesla:cc70 -mp -fast
#CFLAGS = -mp -fast
```

This will add the appropriate flags to compile the code with OpenACC. Now re-compile the code:

```
$ make clean
$ make
```

## Run the GPU-Enabled Code with Increasing Numbers of GPUs

When compiled with OpenACC support, this program splits the matrix elements among OpenMP threads and each OpenMP thread offloads the calculation of its matrix elements to a different GPU.  

Submit the `submit_gpu_01.lsf`, `submit_gpu_02.lsf`, `submit_gpu_03.lsf`, `submit_gpu_04.lsf`, `submit_05.lsf`, and `submit_06.lsf` scripts to run the program with 1, 2, 3, 4, 5, and 6 GPUs:

```
$ bsub submit_gpu_01.lsf
$ bsub submit_gpu_02.lsf
...
```

## Plot the Speedup vs Number of GPUs

The runtime of the 6 GPU-enabled jobs can be found in the files `01_gpu.JOBID`, `02_gpu.JOBID`, `03_gpu.JOBID`, `04_gpu.JOBID`, `05_gpu.JOBID`, and `06_gpu.JOBID`. Use these results to plot the speedup versus number of GPUs relative to the single-OpenMP-thread case. 

The plot should show quite a large speedup! However, you might have decided, and rightfully so, that comparing the GPU-enabled jobs to the single-OpenMP-thread case (single CPU core) isn't really a fair comparison since each GPU has 1000s of compute cores. A more meaningful comparison would be to compare the GPU-enabled runtimes to the 42-OpenMP-thread case. This "full-node" comparison shows the benefit of using the GPUs compared to the full CPU capability of the node (since it uses all 42 physical CPU cores). Now plot the speedup of the 6 GPU-enabled jobs (relative to the 42-OpenMP-thread case) versus the number of GPUs.


