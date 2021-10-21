# Python: Parallel HDF5

Scientific simulations generate large amounts of data on Summit (about 100 Terabytes per day for some applications).
Because of how large some datafiles may be, it is important that writing and reading these files is done as fast as possible.
Less time spent doing input/output (I/O) leaves more time for advancing a simulation or analyzing data.

One of the most utilized file types is the Hierarchical Data Format (HDF), specifically the HDF5 format.
[HDF5](https://www.hdfgroup.org/solutions/hdf5/) is designed to manage large amounts of data and is built for fast I/O processing and storage.
An HDF5 file is a container for two kinds of objects: "datasets", which are array-like collections of data, and "groups", which are folder-like containers that hold datasets and other groups.

There are various tools that allow users to interact with HDF5 data, but we will be focusing on [h5py](https://docs.h5py.org/en/stable/) -- a Python interface to the HDF5 library.
h5py provides a simple interface to exploring and manipulating HDF5 data as if they were Python dictionaries or NumPy arrays.
For example, you can extract specific variables through slicing, manipulate the shapes of datasets, and even write completely new datasets from external NumPy arrays.

Both HDF5 and h5py can be compiled with MPI support, which allows you to optimize your HDF5 I/O in parallel.
MPI support in Python is accomplished through the [mpi4py](https://mpi4py.readthedocs.io/en/stable/) package, which provides complete Python bindings for MPI.
Building h5py against mpi4py allows you to write to an HDF5 file using multiple parallel processes, which can be helpful for users handling large datasets in Python.
h5Py is available after loading the default Python module on either Summit or Ascent, but it has not been built with parallel support.

This hands-on challenge will teach you how to build a personal, parallel-enabled version of h5py and how to write an HDF5 file in parallel using mpi4py and h5py.

Our plan for building parallel h5py is to:

* Create a new virtual environment using conda
* Install mpi4py from source
* Install h5py from source
* Test our build with a Python script

## Setting up the environment

Building h5py from source is highly sensitive to the current environment variables set in your profile.
Because of this, it is extremely important that all the modules and conda environments we plan to load are done in the correct order, so that all the environment variables are set correctly.
First, we will unload all the current modules that you may have previously loaded on Ascent and then immediately load the default modules.
Assuming you cloned the repository in your home directory:

```
$ cd ~/hands-on-with-summit/challenges/Python_Parallel_HDF5
$ source deactivate_envs.sh
$ module purge
$ module load DefApps
```

The `source deactivate_envs.sh` command is only necessary if you already have the Python module loaded.
The script unloads all of your previously activated conda environments, and no harm will come from executing the script if that does not apply to you.

Next, we will load the gnu compiler module (most Python packages assume GCC), hdf5 module (necessary for h5py), and the python module (allows us to create a new conda environment):

```
$ module load gcc
$ module load hdf5
$ module load python
```

Loading the python module puts us in a "base" conda environment, but we need to create a new environment using the `conda create` command:

```
$ conda create -p /ccsopen/proj/<YOUR_PROJECT_ID>/<YOUR_USER_ID>/conda_envs/ascent/h5pympi-ascent python=3.8
```

After following the prompts for creating your new environment, the installation should be successful, and you will see something similar to:

```
Preparing transaction: done
Verifying transaction: done
Executing transaction: done
#
# To activate this environment, use
#
#     $ conda activate /ccsopen/proj/<YOUR_PROJECT_ID>/<YOUR_USER_ID>/conda_envs/ascent/h5pympi-ascent
#
# To deactivate an active environment, use
#
#     $ conda deactivate
```

Due to the specific nature of conda on Ascent, we will be using `source activate` instead of `conda activate` to activate our new environment:

```
$ source activate /ccsopen/proj/<YOUR_PROJECT_ID>/<YOUR_USER_ID>/conda_envs/ascent/h5pympi-ascent
```

The path to the environment should now be displayed in "( )" at the beginning of your terminal lines, which indicate that you are currently using that specific conda environment. 
If you check with `conda env list`, you should see that the `*` marker is next to your new environment, which means that it is currently active:

```
$ conda env list

# conda environments:
#
                      *  /ccsopen/proj/<YOUR_PROJECT_ID>/<YOUR_USER_ID>/conda_envs/ascent/h5pympi-ascent
base                     /sw/ascent/python/3.6/anaconda3/5.3.0
```

## Installing mpi4py

Now that we have a fresh conda environment, we will next install mpi4py from source into our new environment.
To make sure that we are building from source, and not a pre-compiled binary, we will be using pip:

```
$ MPICC="mpicc -shared" pip install --no-binary=mpi4py mpi4py
```

The `MPICC` flag ensures that you are using the correct C wrapper for MPI on the system.
Building from source typically takes longer than a simple `conda install`, so the download and installation may take a couple minutes.
If everything goes well, you should see a "Successfully installed mpi4py" message.

## Installing h5py

Next, we will install h5py from source.
Because h5py depends on NumPy, we will install an optimized version of the NumPy package first using `conda install`:

```
$ conda install -c defaults --override-channels numpy
```

The `-c defaults --override-channels` flags ensure that conda will search for NumPy only on the "defaults" channel.
Installing NumPy in this manner results in an optimized NumPy that is built against linear algebra libraries, which performs operations much faster.

Next, we are finally ready to install h5py from source:

```
$ HDF5_MPI="ON" CC=mpicc pip install --no-binary=h5py h5py
```

The `HDF5_MPI` flag is the key to telling pip to build h5py with parallel support, while the `CC` flag makes sure that we are using the correct C wrapper for MPI.
This installation will take much longer than both the mpi4py and NumPy installations (5+ minutes if the system is slow).
When the installation finishes, you will see a "Successfully installed h5py" message.

## Testing parallel h5py

Now for the fun part, testing to see if our build was truly successful.
We will test our build by trying to write an HDF5 file in parallel using 42 MPI tasks.

First, change directories to your GPFS scratch area and copy over the python and batch scripts:

```
$ cd $MEMBERWORK/<YOUR_PROJECT_ID>
$ cp ~/hands-on-with-summit/challenges/Python_Parallel_HDF5/hello_mpi.py .
$ cp ~/hands-on-with-summit/challenges/Python_Parallel_HDF5/hdf5_parallel.py .
$ cp ~/hands-on-with-summit/challenges/Python_Parallel_HDF5/submit_hello.lsf .
$ cp ~/hands-on-with-summit/challenges/Python_Parallel_HDF5/submit_h5py.lsf .
```

Make sure to edit both "submit_hello.lsf" and "submit_h5py.lsf" to replace any instances of `YOUR_PROJECT_ID` and `YOUR_USER_ID`.

Let's test that mpi4py is working properly first by executing the example Python script "hello_mpi.py".
To do so, we will be submitting a job to the batch queue with "submit_hello.lsf":

```
$ bsub -L $SHELL submit_hello.lsf
```

Once the batch job makes its way through the queue, it will run the "hello_mpi.py" script with 42 MPI tasks.
If mpi4py is working properly, in `mpi4py.<JOB_ID>.out` you should see output similar to:

```
Hello from MPI rank 21 !
Hello from MPI rank 23 !
Hello from MPI rank 28 !
Hello from MPI rank 40 !
Hello from MPI rank 0 !
Hello from MPI rank 1 !
Hello from MPI rank 32 !
.
.
.
```

If you see this, great, it means that mpi4py was built successfully in your environment.

Finally, let's see if we can get these tasks to write to an HDF5 file in parallel using the "hdf5_parallel.py" script:

```python
# hdf5_parallel.py
from mpi4py import MPI
import h5py

comm = MPI.COMM_WORLD      # Use the world communicator
mpi_rank = comm.Get_rank() # The process ID (integer 0-41 for a 42-process job)
mpi_size = comm.Get_size() # Total amount of ranks

with h5py.File('output.h5', 'w', driver='mpio', comm=MPI.COMM_WORLD) as f:
    dset = f.create_dataset('test', (42,), dtype='i')
    dset[mpi_rank] = mpi_rank

comm.Barrier()

if (mpi_rank == 0):
    print('42 MPI ranks have finished writing!')
```

The MPI tasks are going to write to a file named "output.h5", which contains a dataset called "test" that is of size 42 (assigned to the "dset" variable in Python).
Each MPI task is going to assign their rank value to the "dset" array in Python, so we should end up with a dataset that contains 0-41 in ascending order.

Time to execute "hdf5_parallel.py" by submitting "submit_h5py.lsf" to the batch queue:

```
$ bsub -L $SHELL submit_h5py.lsf
```

Provided there are no errors, you should see "42 MPI ranks have finished writing!" in the `h5py.<JOB_ID>.out` output file, and there should be a new file called "output.h5" in your directory.
To see explicitly that the MPI tasks did their job, you can use the `h5dump` command to view the dataset named "test" in output.h5:

```
$ h5dump output.h5

HDF5 "output.h5" {
GROUP "/" {
   DATASET "test" {
      DATATYPE  H5T_STD_I32LE
      DATASPACE  SIMPLE { ( 42 ) / ( 42 ) }
      DATA {
      (0): 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
      (19): 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34,
      (35): 35, 36, 37, 38, 39, 40, 41
      }
   }
}
}
```

If you see the above output, then congratulations you have used one of the fastest computers in the world to write a parallel HDF5 file in Python!

## Additional Resources

* [h5py Documentation](https://docs.h5py.org/en/stable/)
* [mpi4py Documentation](https://mpi4py.readthedocs.io/en/stable/)
* [HDF5 Support Page](https://portal.hdfgroup.org/display/HDF5/HDF5)
