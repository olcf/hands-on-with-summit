# Matrix Multiply on GPU Using hipBLAS

BLAS (Basic Linear Algebra Subprograms) are a set of linear algebra routines that perform basic vector and matrix operations on CPUs. The hipBLAS library includes a similar set of routines that perform basic linear algebra operations on GPUs. 

In this challenge, you will be given a program that initilizes two matrices with random numbers, performs a matrix multiply on the two matrices on the CPU, performs the same matrix multiply on the GPU, then compares the results. The only part of the code that is missing is the call to `hipblasDgemm` that performs the GPU matrix multiply. Your task will be to look up the `hipblasDgemm` routine and add it to the section of the code identified with a `TODO`.

## Add the Call to hipblasDgemm

Before getting started, you'll need to make sure you're in the `GPU_Matrix_Multiply/` directory:

```
$ cd ~/hands-on-with-Frontier-/challenges/GPU_Matrix_Multiply/
```

Look in the code `cpu_gpu_dgemm.cpp` and find the `TODO` section and add in the `hipblasDgemm` call.

> NOTE: You do not need to perform a transpose operation on the matrices, so the `hipblasOperation_t` arguments should be set to `HIPBLAS_OP_N`.

&nbsp;

## Compile the Code

Once you think you've correctly added the hipBLAS routine, try to compile the code.

First, you'll need to make sure your programming environment is set up correctly for this program. You'll need to use the cBLAS library for the CPU matrix multiply (`dgemm`) and the hipBLAS library for the GPU-version (`hipblasDgemm`), so you'll need to load the following modules:

```bash
$ module load PrgEnv-amd              
$ module load openblas
```

Then, try to compile the code:

```bash
$ make
``` 

Did you encounter an error? If so, The compilation errors may assist in identifying the problem. 

## Run the Program

Once you've successfully compiled the code, try running it.

```bash
$ sbatch submit.sbatch
```

If the CPU and GPU give the same results, you will see the message `__SUCCESS__` in the output file. If you do not receive this message, try to identify the problem. As always, if you need help, make sure to ask.
