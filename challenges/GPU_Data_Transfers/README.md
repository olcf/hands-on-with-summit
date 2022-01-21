# Transferring Data Between CPU and GPU

When writing a code to be run on a hybrid compute system (i.e., one with both CPUs and GPUs) such as Summit, you must consider that the CPU and GPU are separate processors with separate memory associated with them. As such, when running a program on this kind of system, control shifts between the CPU and GPU throughout the code and (because of the separate memory) data must be passed back and forth between the two processors as needed.

In this challenge, you will learn how to perform these data transfers with a simple CUDA vector addition program. Your task will be to add in missing arguments to the 3 `cudaMemcpy` API calls (functions used to transfer data between CPU and GPU) so data will be transferred correctly between the CPU (host) and GPU (device). To do so, you will need to look up the `cudaMemcpy` API in NVIDIA's CUDA Runtime API documentation (link below). 

## Basic Outline of the Code

The `vector_addition.cu` code is well documented with comments, but the basic outline of the code is as follows:

* Allocate memory for arrays `A`,  `B`, and `C` on the CPU (commonly called the "host")
* Allocate memory for arrays `d_A`, `d_B`, and `d_C` on the GPU (commonly called the "device")
* Initialize values of arrays `A` and `B` on CPU
* TODO: Transfer data from arrays `A` and `B` (on the CPU) to arrays `d_A` and `d_B` (on the GPU)
* Compute vector addition (`d_C = d_A + d_B`) on the GPU
* TODO: Transfer resulting data from array `d_C` (on the GPU) to array `C` (on the CPU)
* Verify results
* Free GPU memory for arrays `d_A`, `d_B`, and `d_C`
* Free CPU memory for arrays `A`, `B`, and `C`

## Add in the Data Transfers

Before getting started, you'll need to make sure you're in the `GPU_Data_Transfers/` directory:

```
$ cd ~/hands-on-with-summit/challenges/GPU_Data_Transfers/
```

There are two places in the `vector_addition.cu` code (identified with the word `TODO`) where missing arguments will need to be added to the `cudaMemcpy` API calls. Find these two places and add in the missing arguments by looking up the `cudaMemcpy` API call to know which arguments to add. Use [this link](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY_1gc263dbe6574220cc776b45438fc351e8) to the NVIDIA Documentation on the `cudaMemcpy` routine to learn how to use the routine.

> NOTE: You will not need to edit any files other than `vector_addition.cu`.

> NOTE: An Application Programming Interface (API) routine/call should just be thought of as a pre-existing function that can be called from within our program.

## Compile and Run the Program

Once you think you've added the correct lines to the code, try to compile and run it...

First, make sure you have the CUDA Toolkit loaded:

```c
$ module load cuda
``` 

Then, compile the code:

```c
$ make
```

If the code compiles, try to run it as shown below. If not, read the compilation errors and try to determine what went wrong.

```c
$ bsub submit.lsf
```

If the code ran correctly, you will see `__SUCCESS__` along with some other details about how the code was executed. If you don't see this output, try to figure out what went wrong. As always, if you need help, feel free to ask.
