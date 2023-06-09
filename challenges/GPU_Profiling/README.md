# GPU Kernel Profiling Using ROCm

ROCm provides useful profiling tools on Frontier to profile HIP
performance with their ROCProfiler API. We will be using the rocprof command-line tool to generate stats on all kernels being run, the number of times they are run, the total duration and the average duration (in nanoseconds) of the kernel, and the GPU usage percentage. There are numerous ways to use these tools, we encourage you to read more about ROCm Profiling tools
[here](https://docs.amd.com/bundle/ROCProfiler-User-Guide-v5.1/page/Introduction_to_ROCProfiler_User_Guide.html).

In this challenge, you will be profiling two different HIP programs
`matrix_sums_unoptimized.cpp` and `matrix_sums_optimized.cpp`. Each file has two HIP kernels, one that sums the rows of a matrix, and one that sums the columns of the
matrix. The matrix itself is represented as one long array in [row major
order](http://icarus.cs.weber.edu/~dab/cs1410/textbook/7.Arrays/row_major.html). We will
be profiling two different versions of the code. In `matrix_sums_unoptimized.cpp`, the
`row_sums` and `column_sums` kernels uses one thread per row (or one thread per column) to
sum the whole row/column. In `matrix_sums_optimized.cpp`, the `row_sums` kernel is changed
so that it does a [parallel
reduction](https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf) to sum
each row, using one threadblock per row (with 256 threads per block). The `column_sums`
kernel remains the same.

## Step 1: Compile the unoptimized code

First, you'll need to make sure your programming environment is set up correctly for
compiling the code. You need to make sure the ROCm profiling tools are present in the environment. This can be accomplished by loading the modules below.

```bash
$ module load PrgEnv-amd
```

Then you compile the `matrix_sums_unoptimized.cpp` code with

```
$ make unoptimized
```

This will create a binary called `matrix_sums_unoptimized` in the directory.

## Step 2: Run the program

Once you've succesfully compiled, submit the batch script.

```
$ sbatch submit_unoptimized.sbatch
```

If you look inside the batch script, you will see that the program is being run with the
ROCm profiler `rocprof --stats`. This starts the profiler and attaches it to the
program. Check the output file `profiling_unoptimized-JOBID.out` and you will see the
basic profiling output in plain text for the `row_sums` and `column_sums` kernels (scroll
down to get past the loading text). Look at the Kernel Statistics section. Notice the
difference in their duration? The column sum is a lot faster than the row sum. Why is
that? `column_sums` is faster because it takes advantage of _coalesced memory access_. You
can check out [this video](https://www.youtube.com/watch?v=_qSP455IekE) for a brief
explanation of what that is and why it's important.

Also, look at the Memory Operation Statistics sections. Why does copying data
from host to device (HIP memcpy HtoD) take longer?


## Step 3: Compile the optimized code

Now we'll profile the code in `matrix_sums_optimized.cpp`. Look at the `row_sums` kernel
code in `matrix_sums_optimized.cpp` and compare it side-by-side to the code in
`matrix_sums_unoptimized.cpp`. The `column_sums` code is unchanged.

Compile the code in `matrix_sums_optimized.cpp` with:

```
$ make optimized
```

This will create a binary called `sums_optimized` in the directory.

## Step 4: Profile the optimized code

Once you've successfully compiled, run the batch script:

```
$ sbatch submit_optimized.sbatch
```

This also runs the rocprof profiler, same as before. Open the output file
`profiling_optimized-JOBID.out` and check the duration. You can see that the duration
for the `row_sums` is nearly equal to the `column_sums` kernel. Compare this with our
previous output file and you can see it is much faster than the `row_sums` of our previous
code. What causes this?  This is because of the way the `row_sums` was rewritten. It now
takes advantage of memory coalescing and more parallelism compared to the previous
code. The [OLCF CUDA training series](https://www.olcf.ornl.gov/cuda-training-series/) is
an excellent resource if you want to learn more about how and why this works. You can find
slides, recordings, and homework problems to help you learn CUDA programming and optimizing.


# Further information

## HPCToolkit

HPCToolkit is an integrated suite of tools for measurement and analysis of program performance on computers ranging from multicore desktop systems to the nation’s largest supercomputers. HPCToolkit provides accurate measurements of a program’s work, resource consumption, and inefficiency, correlates these metrics with the program’s source code, works with multilingual, fully optimized binaries, has very low measurement overhead, and scales to large parallel systems. HPCToolkit’s measurements provide support for analyzing a program execution cost, inefficiency, and scaling characteristics both within and across nodes of a parallel system. 

You can learn more about the HPCToolkit and other profiling applications on the [Frontier
documentation](https://docs.olcf.ornl.gov/systems/frontier_user_guide.html#getting-started-with-hpctoolkit) page.

## Useful resources

- Coffeebeforearch's video on coalescing in GPUs: https://www.youtube.com/watch?v=_qSP455IekE
- Slides on parallel reduction: https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
- The OLCF CUDA training series: https://www.olcf.ornl.gov/cuda-training-series/
