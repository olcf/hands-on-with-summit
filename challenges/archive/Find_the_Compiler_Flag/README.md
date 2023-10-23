# Find the Missing Compiler Flag
OpenACC is a directive-based approach to programming for GPUs. Instead of using a low-level programming method like HIP/CUDA, where the programmer is responsible for explicitly transferring data between the CPU and GPU and writing GPU kernels, with a directive-based model, the programmer simply adds "hints" within the code which tell the compiler where data transfers should happen and which code sections to offload to the GPU. An additional benefit of this type of GPU programming model is that the code can be compiled for either a CPU or GPU simply by adding or removing compiler flags (whereas a HIP/CUDA code will need to be run on a GPU).

In this challenge, you will need to find the compiler flag that enables GPU support in a simple OpenACC vector addition program. The single `#pragma acc parallel loop` (which is the hint to the compiler) line is the only change needed to make this a GPU code. But without the correct compiler flag, that line will be ignored and a CPU-only executable will be created. 

> NOTE: Compilers flags are optional arguments that can be given to a compiler when compiling a program.

## Step 1: Set Up the Programming Environment

In order to run the provided OpenACC code, we will need to modify our programming environment. 

First, we will change the compiler to gcc:

```bash
$ module load gcc
```

Then, we will load the:

```bash
$ module load cuda
```

## Step 2: Find the Necessary Compiler Flag

Next, you will need to find the GCC compiler flag needed to compile the code with OpenACC-support. To do so, you can either search within the [Frontier User Guide](https://docs.olcf.ornl.gov/systems/frontier_user_guide.html#), within the [GCC Wiki](https://gcc.gnu.org/wiki/HomePage), or just Google "GCC OpenACC compiler flag". Being able to use on-line resources to find e.g., compiler flags is often a necessary task in HPC.

> NOTE: Compiler flags differ between different compilers so make sure you find the correct flag for the **GCC compiler**.

## Step 3: Add the Compiler Flag to the Makefile and Compile

First, make sure you're in the `Find_the_Compiler_Flag` directory:

```bash
$ cd ~/hands-on-with-Frontier-/challenges/Find_the_Compiler_Flag
```

Ok, if you haven't done so already, go find the compiler flag...

Ok, now that you think you found the correct compiler flag, add it to the end of the `CFLAGS = ` line in the Makefile. Then, compile the code:

```bash
$ make
```

If you added the correct flag, you should see evidence of it in the output from the compiler (i.e., the acc and copyout/copyin lines):

```
main:
     24, Generating Tesla code
         25, #pragma acc loop gang, vector(128) /* blockIdx.x threadIdx.x */
     24, Generating implicit copyout(C[:1048576])
         Generating implicit copyin(B[:1048576],A[:1048576])
```

## Step 4: Run the Program

Now, test that you have correctly compiled the code with OpenACC-support by launching the executable on a compute node. To do so, issue the following command:

```
$ sbatch submit.sbatch
```

> NOTE: The submit.sbatch script requests access to 1 compute node for 10 minutes and launches the executable on that compute node using the job launcher, `srun`.


Once the job is complete, you can confirm that it gave the correct results by looking for `__SUCCESS__` in the output file, `add_vec_acc-JOBID.out`, where JOBID will be the unique number associated with your job. 

But did you run on the CPU or GPU? An easy way to tell is using ????????'s profiler, ????????. This was included in the `srun` command so if you ran on the GPU, you should also see output from the profiler as shown below:

```
==6163== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   62.26%  363.49us         2  181.74us  181.60us  181.89us  [CUDA memcpy HtoD]
                   31.17%  181.98us         1  181.98us  181.98us  181.98us  [CUDA memcpy DtoH]
                    6.57%  38.368us         1  38.368us  38.368us  38.368us  main_24_gpu
```

If you need help, don't be afraid to ask. If you were successful, congratulations! You just ran a program on a GPU in one of the fastest supercomputers in the world!


