# GPU Kernel Profiling Using Nsight Systems

NVIDIA provides a couple of useful profiling tools on Summit to profile CUDA
performance. Nsight Systems and Nsight
Compute. Typically, you first use Nsight Systems to profile your whole program to identify
any bottlenecks. You use Nsight Compute to profile the kernels in more detail. We will be
focusing on Nsight Systems in this challenge to time the CUDA kernels. You can read more
about the use of both tools on Summit
[here](https://docs.olcf.ornl.gov/systems/summit_user_guide.html#profiling-gpu-code-with-nvidia-developer-tools).

In this challenge, you will be profiling two different CUDA programs
`matrix_sums_unoptimized.cu` and `matrix_sums_optimized.cu`. Each file has two CUDA
kernels, one that sums the rows of a matrix, and one that sums the columns of the
matrix. The matrix itself is represented as one long array in [row major
order](http://icarus.cs.weber.edu/~dab/cs1410/textbook/7.Arrays/row_major.html). We will
be profiling two different versions of the code. In `matrix_sums_unoptimized.cu`, the
`row_sums` and `column_sums` kernels uses one thread per row (or one thread per column) to
sum the whole row/column. In `matrix_sums_optimized.cu`, the `row_sums` kernel is changed
so that it does a [parallel
reduction](https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf) to sum
each row, using one threadblock per row (with 256 threads per block). The `column_sums`
kernel remains the same.

## Step 1: Compile the unoptimized code

First, you'll need to make sure your programming environment is set up correctly for
compiling the code. You need to make sure CUDA, gcc and the profiling tools are present
in the environment.

```
$ module load gcc
$ module load cuda
$ module load nsight-systems
```

Then you compile the `matrix_sums_unoptimized.cu` code with

```
$ make unoptimized
```

This will create a binary called `matrix_sums_unoptimized` in the directory.

## Step 2: Run the program

Once you've succesfully compiled, submit the batch script.

```
$ bsub submit_unoptimized.lsf
```

If you look inside the batch script, you will see that the program is being run with the
Nsight Systems profiler `nsys profile`. This starts the profiler and attaches it to the
program. Check the output file `profiling_output_unoptimized.<jobid>` and you will see the
basic profiling output in plain text for the `row_sums` and `column_sums` kernels (scroll
down to get past the loading text). Look at the CUDA Kernel Statistics section. Notice the
difference in their duration? The column sum is a lot faster than the row sum. Why is
that? `column_sums` is faster because it takes advantage of _coalesced memory access_. You
can check out [this video](https://www.youtube.com/watch?v=_qSP455IekE) for a brief
explanation of what that is and why it's important.

Also, look at the Memory Operation Statistics sections. Why does copying data
from host to device (CUDA memcpy HtoD) take longer?


## Step 3: Compile the optimized code

Now we'll profile the code in `matrix_sums_optimized.cu`. Look at the `row_sums` kernel
code in `matrix_sums_optimized.cu` and compare it side-by-side to the code in
`matrix_sums_unoptimized.cu`. The `column_sums` code is unchanged.

Compile the code in `matrix_sums_optimized.cu` with:

```
$ make optimized
```

This will create a binary called `sums_optimized` in the directory.

## Step 4: Profile the optimized code

Once you've successfully compiled, run the batch script:

```
$ bsub submit_optimized.lsf
```

This also runs the Nsight Systems profiler, same as before. Open the output file
`profiling_output_optimized.<jobid>` and check the duration. You can see that the duration
for the `row_sums` is nearly equal to the `column_sums` kernel. Compare this with our
previous output file and you can see it is much faster than the `row_sums` of our previous
code. What causes this?  This is because of the way the `row_sums` was rewritten. It now
takes advantage of memory coalescing and more parallelism compared to the previous
code. The [OLCF CUDA training series](https://www.olcf.ornl.gov/cuda-training-series/) is
an excellent resource if you want to learn more about how and why this works. You can find
slides, recordings, and homework problems to help you learn CUDA programming and optimizing.


# Further information

## Profiling GUI Tools

The profiler also creates a `.qdrep` file. This can be opened in the Nsight Systems GUI
tool (you will need to download and install it on your desktop, and download the `.qdrep`
file to your local machine via `scp`) to provide a visualization of the program's run. You
can find more instructions on how to get and use the GUI tool in the [Summit
documentation](https://docs.olcf.ornl.gov/systems/summit_user_guide.html#optimizing-and-profiling).

## Useful resources

- Coffeebeforearch's video on coalescing in GPUs: https://www.youtube.com/watch?v=_qSP455IekE
- Slides on parallel reduction: https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
- The OLCF CUDA training series: https://www.olcf.ornl.gov/cuda-training-series/
