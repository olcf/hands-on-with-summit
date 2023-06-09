# Basic Workflow to Run a Job on Frontier

The basic workflow for running programs on HPC systems is 1) set up your programming environment - i.e., the software you need, 2) compile the code - i.e., turn the human-readable programming language into machine code, 3) request access to one or more compute nodes, and 4) launch your executable on the compute node(s) you were allocated. In this challenge, you will perform these basic steps to see how it works.

For this challenge you will compile and run a vector addition program written in C. It takes two vectors (A and B), adds them element-wise, and writes the results to vector C:

```c
// Add vectors (C = A + B)
for(int i=0; i<N; i++)
{
    C[i] = A[i] + B[i];
}
```

## Step 1: Setting Up Your Programming Environment
Many software packages and scientific libraries are pre-installed on Frontier for users to take advantage of. Several packages are loaded by default when a user logs in to the system and additional packages can be loaded using environment modules. To see which packages are currently loaded in your environment, run the following command:

```
$ module list
``` 

> NOTE: The `$` in the command above represents the "command prompt" for the bash shell and is not part of the command that needs to be executed.

For this example program, we will use the AMD compiler. To use the AMD compiler, load the AMD programming environment by issuing the following command:

```
$ module load PrgEnv-amd
```

## Step 2: Compile the Code

Now that you've set up your programming environment for the code used in this challenge, you can go ahead and compile the code. First, make sure you're in the `Basic_Workflow/` directory:

```
$ cd ~/hands-on-with-Frontier-/challenges/Basic_Workflow
```

> NOTE: The path above assumes you cloned the repo in your `/ccs/home/username` directory.

Then compile the code. To do so, you'll use the provided `Makefile`, which is a file containing the set of commands to automate the compilation process. To use the `Makefile`, issue the following command:

```
$ make
```

Based on the commands contained in the `Makefile`, an executable named `run` will be created.

## Steps 3-4: Request Frontier's to Compute Nodes and Run the Program

In order to run the executable on Frontier's compute nodes, you need to request access to a compute node and then launch the job on the node. The request and launch can be performed using the single batch script, `submit.sbatch`. If you open this script, you will see several lines starting with `#SBATCH`, which are the commands that request a compute node and define your job (i.e., give me 1 compute node for 10 minutes, charge project `PROJID` for the time, and name the job and output file `add_vec_cpu`). You will also see a `srun` command within the script, which launches the executable (`run`) on the compute node you were given. 

The flags given to `srun` define the resources (i.e., number of processors and number of processors per node) available to your program and the processes/threads you want to run on those resources. 

> (For more information on using the `srun` job launcher, please see challenge [srun\_Job\_Launcher](../srun_Job_Launcher))

&nbsp;

To submit and run the job, issue the following command:

```bash
$ sbatch submit.sbatch
```

## Monitoring Your Job

Now that the job has been submitted, you can monitor its progress. Is it running yet? Has it finished? To find out, you can issue the command 

```bash
$ squeue -u USERNAME
```

where `USERNAME` is your username. This will show you the state of your job to determine if it's running, eligible (waiting to run), or blocked. When you no longer see your job listed with this command, you can assume it has finished (or crashed). Once it has finished, you can see the output from the job in the file named `add_vec_cpu-JOBID.out`, where `JOBID` is the unique ID given to you job when you submitted it. 
List your directoy to verify that `add_vec_cpu-JOBID.out` is present by doing: 
```
$ ls
```
You can confirm that it gave the correct results by looking for `__SUCCESS__` in the output file. To see the contents of the file, do: 

```
cat add_vec_cpu-JOBID.out
```

