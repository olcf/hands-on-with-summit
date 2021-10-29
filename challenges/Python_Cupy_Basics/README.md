# Python: CuPy Basics

GPU computing has become a big part of the data science landscape, as array operations with NVIDIA GPUs can provide considerable speedups over CPU computing.
Although GPU computing on Summit is often utilized in codes that are written in Fortran and C, GPU-related Python packages are quickly becoming popular in the data science community.
One of these packages is [CuPy](https://cupy.dev/), a NumPy/SciPy-compatible array library accelerated with NVIDIA CUDA.

CuPy is a library that implements NumPy arrays on NVIDIA GPUs by utilizing CUDA Toolkit libraries like cuBLAS, cuRAND, cuSOLVER, cuSPARSE, cuFFT, cuDNN and NCCL.
Although optimized NumPy is a significant step up from Python in terms of speed, performance is still limited by the CPU (especially at larger data sizes) -- this is where CuPy comes in.
Because CuPy's interface is nearly a mirror of NumPy, it acts as a replacement to run existing NumPy/SciPy code on NVIDIA CUDA platforms, which helps speed up calculations further.
CuPy supports most of the array operations that NumPy provides, including array indexing, math, and transformations.
Most operations provide an immediate speed-up out of the box, and some operations are sped up by over a factor of 100 (see CuPy benchmark timings below, from the [Single-GPU CuPy Speedups](https://medium.com/rapids-ai/single-gpu-cupy-speedups-ea99cbbb0cbb) article).

<p align="center" width="100%">
    <img width="50%" src="images/cupy_chart.png">
</p>

Because each Ascent compute node has 6 NVIDIA V100 GPUs, we will be able to take full advantage of CuPy's capabilities on the system, providing significant speedups over NumPy-written code.

In this challenge, you will:

* Learn how to install CuPy into a custom conda environment
* Learn the basics of CuPy
* Apply what you've learned in a debugging challenge
* Compare speeds to NumPy on Ascent (bonus)

## Installing CuPy

First, we will unload all the current modules that you may have previously loaded on Ascent and then immediately load the default modules:

```
$ cd ~/hands-on-with-summit/challenges/Python_Cupy_Basics
$ source deactivate_envs.sh
$ module purge
$ module load DefApps
```

The `source deactivate_envs.sh` command is only necessary if you already have the Python module loaded.
The script unloads all of your previously activated conda environments, and no harm will come from executing the script if that does not apply to you.

Next, we will load the gnu compiler module (most Python packages assume GCC), cuda module (necessary for CuPy), and the python module (allows us to create a new conda environment):

```
$ module load gcc/7.4.0
$ module load cuda/11.0.2
$ module load python
```

Loading the python module puts us in a "base" conda environment, but we need to create a new environment using the `conda create` command:

```
$ conda create -p /ccsopen/home/<YOUR_USER_ID>/.conda/envs/cupy-ascent python=3.9
```

> NOTE: As noted in [Conda Basics](../Python_Conda_Basics), it is highly recommended to create new environments in the "Project Home" directory.
> However, due to the limited disk quota and potential number of training participants on Ascent, we will be creating our environment in the "User Home" directory.

After following the prompts for creating your new environment, the installation should be successful, and you will see something similar to:

```
Preparing transaction: done
Verifying transaction: done
Executing transaction: done
#
# To activate this environment, use
#
#     $ conda activate /ccsopen/home/<YOUR_USER_ID>/.conda/envs/cupy-ascent
#
# To deactivate an active environment, use
#
#     $ conda deactivate
```

Due to the specific nature of conda on Ascent, we will be using `source activate` instead of `conda activate` to activate our new environment:

```
$ source activate /ccsopen/home/<YOUR_USER_ID>/.conda/envs/cupy-ascent
```

The path to the environment should now be displayed in "( )" at the beginning of your terminal lines, which indicates that you are currently using that specific conda environment.
If you check with `conda env list`, you should see that the `*` marker is next to your new environment, which means that it is currently active:

```
$ conda env list

# conda environments:
#
                      *  /ccsopen/home/<YOUR_USER_ID>/.conda/envs/cupy-ascent
base                     /sw/ascent/python/3.6/anaconda3/5.3.0
```

CuPy depends on NumPy, so let's install an optimized version of NumPy into our fresh conda environment:

```
$ conda install -c defaults --override-channels numpy
```

After following the prompts, NumPy and its linear algebra dependencies should successfully install.

Next, we will install SciPy.
SciPy is an optional dependency, but it would allow us to use the additional SciPy-based routines in CuPy:

```
$ conda install scipy
```

Finally, we will install CuPy from source into our environment.
To make sure that we are building from source, and not a pre-compiled binary, we will be using pip:

```
$ CUDA_PATH="${CUDAPATH}" CC=gcc NVCC=nvcc pip install --no-binary=cupy cupy
```

The `CUDA_PATH` flag makes sure that we are using the correct path set by the `cuda/11.0.2` module, while the `CC` and `NVCC` flags ensure that we are passing the correct wrappers.
This installation takes, on average, 20 minutes to complete (due to building everything from scratch), so don't panic if it looks like the install timed-out.
Eventually you should see output similar to:

```
Successfully installed cupy-9.5.0 fastrlock-0.6
```

Congratulations, you just installed CuPy on Ascent!

## Getting Started With CuPy

Before we start testing the CuPy scripts provided in this repository, let's go over some of the basics.
The developers provide a great introduction to using CuPy in their user guide under the [CuPy Basics](https://docs.cupy.dev/en/stable/user_guide/basic.html) section.
We will be following this walkthrough on Ascent.
This is done to illustrate the basics, but participants should **NOT** explicitly follow along (as resources are limited on Ascent and interactive jobs will clog up the queue).
The syntax below assumes being in a Python shell with access to 4 GPUs.

> NOTE: Assuming you are continuing from the previous section, you do not need to load any modules.
> However, if you logged out after finishing the previous section, you must `module load python` and activate your CuPy conda environment before moving on.

As is the standard with NumPy being imported as "np", CuPy is often imported in a similar fashion:

```python
>>> import numpy as np
>>> import cupy as cp
```

Similar to NumPy arrays, CuPy arrays can be declared with the `cupy.ndarray` class.
NumPy arrays will be created on the CPU (the "host"), while CuPy arrays will be created on the GPU (the "device"):

```python
>>> x_cpu = np.array([1,2,3])
>>> x_gpu = cp.array([1,2,3])
```

Manipulating a CuPy array can also be done in the same way as manipulating NumPy arrays:

```python
>>> x_cpu*2.
array([2., 4., 6.])
>>> x_gpu*2.
array([2., 4., 6.])
>>> l2_cpu = np.linalg.norm(x_cpu)
>>> l2_gpu = cp.linalg.norm(x_gpu)
>>> print(l2_cpu,l2_gpu)
3.7416573867739413 3.7416573867739413
```

Useful functions for initializing arrays like `np.linspace`, `np.arange`, and `np.zeros` also have a CuPy equivalent:

```python
>>> cp.zeros(3)
array([0., 0., 0.])
>>> cp.linspace(0,10,11)
array([ 0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9., 10.])
>>> cp.arange(0,11,1)
array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10])
```

CuPy has a concept of a "current device", which is the current activated GPU device that will operate on an array or where future arrays will be allocated.
Most of the time, if not explicitly declared or switched, the initial default device will be GPU 0.
To find out what device a CuPy array is allocated on, you can call the `cupy.ndarray.device` attribute:

```python
>>> x_gpu.device
<CUDA Device 0>
```
To get a total number of devices that you can access, use the `getDeviceCount` function:

```python
>>> cp.cuda.runtime.getDeviceCount()
4
```

Unless run with a specific `jsrun -g <NUMBER_OF_GPUS>` command, the launch node has 4 GPUs that CuPy can find.
The current device can be switched using `cupy.cuda.Device(<DEVICE_ID>).use()`:

```python
>>> cp.cuda.Device(1).use()
>>> x_gpu_1 = cp.array([1, 2, 3, 4, 5])
>>> x_gpu_1.device
<CUDA Device 1>
```

Similarly, you can temporarily switch to a device using the `with` context:

```python
>>> cp.cuda.Device(0).use()
>>> with cp.cuda.Device(3):
...    x_gpu_3 = cp.array([1, 2, 3, 4, 5])
...
>>> x_gpu_0 = cp.array([1, 2, 3, 4, 5])
>>> x_gpu_0.device
<CUDA Device 0>
>>> x_gpu_3.device
<CUDA Device 3>
```

Trying to perform operations on an array stored on a different GPU will result in an error:

```python
>>> with cp.cuda.Device(0):
...    x_gpu_0 = cp.array([1, 2, 3, 4, 5]) # create an array in GPU 0
...
>>> with cp.cuda.Device(1):
...    x_gpu_0 * 2  # ERROR: trying to use x_gpu_0 on GPU 1
...
Traceback (most recent call last):
ValueError: Array device must be same as the current device: array device = 0 while current = 1
```

To solve the above error, we must transfer `x_gpu_0` to "Device 1".
A CuPy array can be transferred to a specific GPU using the `cupy.asarray()` function while on the specific device:

```python
>>> with cp.cuda.Device(1):
...    cp.asarray(x_gpu_0) * 2  # fixes the error, moves x_gpu_0 to GPU 1
...
array([ 2,  4,  6,  8, 10])
```

A NumPy array on the CPU can also be transferred to a GPU using the same `cupy.asarray()` function:

```python
>>> x_cpu = np.array([1, 1, 1]) # create an array on the CPU
>>> x_gpu = cp.asarray(x_cpu)  # move the CPU array to the current device
>>> x_gpu
array([1, 1, 1])
```

To transfer from a GPU back to the CPU, you use the `cupy.asnumpy()` function instead:

```python
>>> x_gpu = cp.zeros(3)  # create an array on the current device
>>> x_cpu = cp.asnumpy(x_gpu)  # move the GPU array to the CPU
>>> x_cpu
array([ 0., 0., 0.])
```

Associated with the concept of current devices are current "streams".
In CuPy, all CUDA operations are enqueued onto the current stream, and the queued tasks on the same stream will be executed in serial (but asynchronously with respect to the CPU).
This can result in some GPU operations finishing before some CPU operations.
As CuPy streams are out of the scope of this challenge, you can find additional information in the [CuPy User Guide](https://docs.cupy.dev/en/stable/user_guide/index.html).

Congratulations, you now know some of the basics of CuPy!

Now let's apply what you've learned.

## Data Transfer Debugging Challenge

Before asking for a compute node, let's change into our scratch directory and copy over the relevant files.

```
$ cd $MEMBERWORK/<YOUR_PROJECT_ID>
$ mkdir cupy_test
$ cd cupy_test
$ cp ~/hands-on-with-summit/challenges/Python_Cupy_Basics/*.py .
$ cp ~/hands-on-with-summit/challenges/Python_Cupy_Basics/*.lsf .
```

When a kernel call is required in CuPy, it compiles a kernel code optimized for the shapes and data types of given arguments, sends it to the GPU device, and executes the kernel. 
Due to this, CuPy runs slower on its initial execution.
This slowdown will be resolved at the second execution because CuPy caches the kernel code sent to GPU device.
By default, the compiled code is cached to `$(HOME)/.cupy/kernel_cache` directory, which the compute nodes will not be able to access.
We will change it to our scratch directory:

```
$ export CUPY_CACHE_DIR="/gpfs/wolf/scratch/<YOUR_USER_ID>/<YOUR_PROJECT_ID>/.cupy/kernel_cache"
```

Now, it's time to dive into `data_transfer.py`:

```python
# data_transfer.py
import numpy as np
import cupy as cp

# Initialize the array x_gpu_0 on GPU 0
x_gpu_0 = cp.arange(10)
cp.cuda.Stream.null.synchronize() # Waits for GPU 0 to finish
print(x_gpu_0.device, 'done')

# Modify a copy of x_gpu_0 on GPU 1 (must send data to GPU 1)
with cp.cuda.Device(1):
        x_gpu_1 = x_gpu_0 # TO-DO
        x_gpu_1 = x_gpu_1**2
        cp.cuda.Stream.null.synchronize() # Waits for GPU 1 to finish
        print(x_gpu_1.device, 'done')

# Modify a copy of x_gpu_0 on GPU 2 (must send data to GPU 2)
with cp.cuda.Device(2):
        x_gpu_2 = x_gpu_0 # TO-DO
        x_gpu_2 = x_gpu_2**3
        cp.cuda.Stream.null.synchronize() # Waits for GPU 2 to finish
        print(x_gpu_2.device, 'done')

# Sum all arrays on the CPU (must send arrays to the CPU)
x_cpu = x_gpu_0 +  x_gpu_1 + x_gpu_2 # TO-DO
print('Finished computing on the CPU\n')

# Summary of our results
print('Results:')
print(x_gpu_0.device, ':', x_gpu_0)
print(x_gpu_1.device, ':', x_gpu_1)
print(x_gpu_2.device, ':', x_gpu_2)
print('CPU: ', x_cpu)
```

The goal of the above script is to calculate `x + x^2 + x^3` after calculating `x^2` and `x^3` on separate GPUs.
To do so, the working script initializes `x` on GPU 0, then make copies on both GPU 1 and GPU 2.
After all of the GPUs finish their calculations, the CPU then computes the final sum of all the arrays.
However, running the above script will result in errors, so it is your mission to figure out how to fix it.
Specifically, there are three lines that need fixing in this script (marked by the "TO-DO" comments).

Your challenge is to apply the necessary function calls on the three "TO-DO" lines to transfer the data between the GPUs and CPUs properly.
Some of the questions to help you: What function do I use to pass arrays to a GPU? What function do I use to pass arrays to a CPU?
If you're having trouble, you can check `data_transfer_solution.py`.

To do this challenge:

1. Determine the missing functions on the three "TO-DO" lines.
2. Use your favorite editor to enter the missing functions into `data_transfer.py`. For example:

    ```
    $ vi data_transfer.py
    ```

3. Submit a job:

    ```
    $ bsub -L $SHELL submit_data.lsf
    ```

4. If you fixed the script, you should see the below output in `cupy_xfer.<JOB_ID>.out` after the job completes:

    ```python
    <CUDA Device 0> done
    <CUDA Device 1> done
    <CUDA Device 2> done
    Finished computing on the CPU

    Results:
    <CUDA Device 0> : [0 1 2 3 4 5 6 7 8 9]
    <CUDA Device 1> : [ 0  1  4  9 16 25 36 49 64 81]
    <CUDA Device 2> : [  0   1   8  27  64 125 216 343 512 729]
    CPU:  [  0   3  14  39  84 155 258 399 584 819]
    ```

If you got the script to successfully run, then congratulations!

## Bonus: NumPy Speed Comparison

Now that you know how to use CuPy, that brings us to seeing the actual benefits that CuPy provides for large datasets.
More specifically, let's see how much faster CuPy can be than NumPy on Ascent.
You won't need to fix any errors; this is mainly a demonstration on what CuPy is capable of.

There are a few things to consider when running on GPUs, which also apply to using CuPy:

* Higher precision means higher cost (time and space)
* The structuring of your data is important
* The larger the data, the better for GPUs (but needs careful planning)

These points are explored in the script `timings.py`.
Let's inspect the script:

```python
# timings.py
import cupy as cp
import numpy as np
import time as tp

A      = np.random.rand(3000,3000) # NumPy rand
G      = cp.random.rand(3000,3000) # CuPy rand
G32    = cp.random.rand(3000,3000,dtype=cp.float32) # Create float32 matrix instead of float64 (default)
G32_9k = cp.random.rand(9000,1000,dtype=cp.float32) # Create float32 matrix of a different shape

t1 = tp.time()
np.linalg.svd(A) # NumPy Singular Value Decomposition
t2 = tp.time()
print("CPU time: ", t2-t1)

t3 = tp.time()
cp.linalg.svd(G) # CuPy Singular Value Decomposition
cp.cuda.Stream.null.synchronize() # Waits for GPU to finish
t4 = tp.time()
print("GPU time: ", t4-t3)

t5 = tp.time()
cp.linalg.svd(G32)
cp.cuda.Stream.null.synchronize()
t6 = tp.time()
print("GPU float32 time: ", t6-t5)

t7 = tp.time()
cp.linalg.svd(G32_9k)
cp.cuda.Stream.null.synchronize()
t8 = tp.time()
print("GPU float32 restructured time: ", t8-t7)
```

This script times the decomposition of a matrix with 9 million elements across four different methods.
First, NumPy is timed for a 3000x3000 dimension matrix.
Then, a 3000x3000 matrix in CuPy is timed.
As you will see shortly, the use of CuPy will result in a major performance boost when compared to NumPy, even though the matrices are structured the same way.
This is improved upon further by switching the data type to `float32` from `float64` (the default).
Lastly, a 9000x1000 matrix is timed, which contains the same number of elements as the original matrix, just rearranged.
Although you may not expect it, the restructuring results in a big performance boost as well.

Let's see the boosts explicitly by running the `timings.py` script.
To do so, you must submit `submit_timings.lsf` to the queue:

```
$ bsub -L $SHELL submit_timings.lsf
```

After the job completes, in `cupy_timings.<JOB_ID>.out` you will see something similar to:

```python
CPU time:  21.632022380828857
GPU time:  11.382664203643799
GPU float32 time:  4.066986799240112
GPU float32 restructured time:  0.8666532039642334
```

The exact numbers may be slightly different, but you should see a speedup factor of approximately 2 or better when comparing "GPU time" to "CPU time".
Switching to `float32` was easier on memory for the GPU, which improved the time further.
Things are even better when we look at "GPU float32 restructured time", which represents an additional factor of 4 speedup when compared to "GPU float32 time".
Overall, using CuPy and restructuring the data led to a speedup factor of >20 when compared to traditional NumPy!
This factor would diminish with smaller datasets, but represents what CuPy is capable of.

You have now discovered what CuPy can provide!
Now you can try speeding up your own codes by swapping CuPy and NumPy where you can.

## Additional Resources

* [CuPy User Guide](https://docs.cupy.dev/en/stable/user_guide/index.html)
* [CuPy Website](https://cupy.dev/)
* [CuPy API Reference](https://docs.cupy.dev/en/stable/reference/index.html)
