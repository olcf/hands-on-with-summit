Slurm
-----

**Goals:** 

* Familiarize yourself with batch scripts
* Understand how to modify and submit an existing batch script
* Understand how to run a job with Slurm
* Explore job layout on the node.
* Understand how to query the queue for your jobs status and read the results


Frontier uses SchedMD's Slurm Workload Manager for scheduling and managing jobs. Slurm maintains similar functionality to other schedulers such as IBM's LSF, but provides unique control of Frontier's resources through custom commands and options specific to Slurm. 

Slurm documentation for each command is available via the ``man`` utility, and on the web at [https://slurm.schedmd.com/man_index.html](https://slurm.schedmd.com/man_index.html). Additional documentation is available at [https://slurm.schedmd.com/documentation.html](https://slurm.schedmd.com/documentation.html)

Some common Slurm commands are summarized in the table below. 


| Command      | Action/Task                                    | 
|:-------:     | :----------                                    | 
| ``squeue``   | Show the current queue                         | 
| ``sbatch``   | Submit a batch script                          | 
| ``salloc``   | Submit an interactive job                      | 
| ``srun``     | Launch a parallel job                          | 
| ``sacct``    | View accounting information for jobs/job steps | 
| ``scancel``  | Cancel a job or job step                       | 
| ``scontrol`` | View or modify job configuration.              | 

This challenge will guide you through using `sbatch` the command to send a job to the scheduler, `srun` the parallel job launcher and `squeue`, the command that shows the jobs that a queued to run. We will be submitting the jobs via batch scripts that allow us to take advantage of the scheduler to manage the workload. Let's start by first setting up our test code and then learning how to run it with a batch script.  

Compiling the Test code
-----------------------
We will use a code called `hello_mpi_omp` written by Tom Papatheodore as our test example. This code's output will display where each process runs on the compute node. 

To begin, make sure you are in the directory for the challenge by doing: 

```
cd ~/hands-on-with-Frontier-/challenges/srun_Job_Launcher

```

Do ``ls`` to verify that you see `hello_mpi_omp.c`, `makefile` and `submit.sl` listed. 

We compile the code using a makefile, which is a file that specifies how to compile the program. If you are curious, you may view the makefile by doing `vi Makefile`, but you do not need to understand that file to achieve the goals of this exercise. 

We will use the default programming environment on Frontier to compile this code, which means using the Cray programming environment and Cray-MPICH for MPI. On Frontier these are set up when you login. If running this tutorial on other machines, you would need to use their documentation to learn how to setup a programming environment to support MPI and OpenMP.

To use the Makefile to compile the code on Frontier, do:
```
make
```
If all goes right, you will see that this produces an executable file called `hello_mpi_omp`. You may use `ls` to verify that the file is present. If you don't see it retrace your steps to find what you missed. You may also ask an instructor. 

Now that we have an executable to run, let’s modify and use a batch script to run it!


Batch Scripts
-------------
The most common way to interact with the batch system is via batch scripts. A batch script is simply a shell script with added directives to request various resources from or provide certain information to the scheduling system.  Aside from these directives, the batch script is simply the series of commands needed to set up and run your job.

To submit a batch script, use the command ``sbatch submit.sl``, but don't do that just yet, because you will need to customize the example batch script, `submit.sl` first. 

The example batch script, `submit.sl` looks like this:

```
#!/bin/bash
#SBATCH -A TRN###
#SBATCH -J srun_<enter your name here>
#SBATCH -o %x-%j.out
#SBATCH -t 10:00
#SBATCH -p batch
#SBATCH -N 1

# number of OpenMP threads
export OMP_NUM_THREADS=1

# jsrun command to modify 
srun -N 1 -n 1 -c 1 ./hello_mpi_omp

```

In the script, Slurm directives are preceded by ``#SBATCH``, making them appear as comments to the shell. Slurm looks for these directives through the first non-comment, non-whitespace line. Options after that will be ignored by Slurm (and the shell).


| Line | Description                                                                                     |
| :--: | :----------                                                                                     |
|    1 | Shell interpreter line                                                                          |
|    2 | OLCF project to charge                                                                          |
|    3 | Job name                                                                                        |
|    4 | Job standard output file (``%x`` will be replaced with the job name and ``%j`` with the Job ID) |
|    5 | Walltime requested (in ``MM:SS`` format). See the table below for other formats.                |
|    6 | Partition (queue) to use                                                                        |
|    7 | Number of compute nodes requested                                                               |
|    8 | Blank line                                                                                      |
|    9 | Comment                                                                                         |
|   10 | Sets the number of OpenMP threads                                                               |
|   11 | Blank line                                                                                      |
|   12 | Comment                                                                                         |
|   13 | Run the job ( add layout details )                                                              |


Next, let's modify your copy of this batch script to give you practice. 
Open the batch script with vi (or your favored text editor). To use vi do the following:

```
vi submit.sl

```

Hit `esc`, and then type `i` to put vi in insert mode, use the explanations in the example above to make the following modifications to the batch script:
1. Change the project ID to the project ID for this tutorial.
2. Customize your job's name.
3. Change the time from 10 minutes to 8 minutes.

When you are finished, come out of vi insert mode by hitting `esc` then type `:wq` and hit `return` to save your file and quit vi.  

Submit the batch script to run by doing:

```
sbatch submit.sl
```

To see what state your job is in use: 
```
squeue -u <your_user_id>
````
The sections below will help you understand how to read the output that is given.


Job States
----------

A job will transition through several states during its lifetime. Common ones include:

| Code | State      | Description                                                                    |
| :--- |:-----      | :----------                                                                    | 
| CA    | Canceled   | The job was canceled (could've been by the user or an administrator)         |
| CD    | Completed  | The job completed successfully (exit code 0)                                  |
| CG    | Completing | The job is in the process of completing (some processes may still be running) |
| PD    | Pending    | The job is waiting for resources to be allocated                              |
| R     | Running    | The job is currently running                                                  |


Did you see that your job was queued and why it was not running yet? It’s possible that your job could have run before your query was sent, so there might not be an entry for it in the queue. 
The other way to check if the job ran, is to see if there is an output file in your directory.

Typing `ls` will list all the files in your current directory and including the output file if it exists.  

The filename will be composed of the name that your chose for your job in the batch script followed by the job ID number from the queue followed by `.out`. My output files look like this:
```
srun_SPK-397601.out
```

Srun Results Example
--------------------------
Let's examine the output of your job and use that example to begin to understand how to use srun commands to organize work on the compute node. To interpret the results, you need to understand some basics of the node hardware, and the parallel programming models utilized by hello_mpi_omp, called MPI and OpenMP.

*Compute Node*

The compute nodes are composed of a CPU made of several hardware cores that have a few hardware threads each. Most modern HPC nodes also have GPUs, but we will not focus on those yet. 

Below is a picture of the Frontier compute node. 


<br>
<center>
<img src="images/Frontier-Visualizer-NodeLegend.png" style="width:80%">
</center>
<br>

The blue portion of the pictured node is the CPU. You can see that it has several rows of cores each represented by a small blue box. Each core has two hardware threads, labeled with numbers. Each thread can do an independent task.  The red boxes to the right are the GPUs, we will not talk about scheduling them in this tutorial, but the reason that they are part of most modern supercomputers is that they each have thousands of channels which enable them to do streams of 1000s of independent operations. 

 If you are running this tutorial on a different computer, the details of the node may look a little different, but the basic elements of cores and hardware threads will be similar for the CPU.   

*Programming Models* 

To organize work in parallel, hello_mpi_omp uses MPI tasks and OpenMP threads. These are specified by the program, and each does a specific task as set by the programmer. In the case of our hello_mpi_omp program, each MPI task gets the name of the node running the code and organizes its associated OpenMP processes to store their process IDs and the ID of the hardware thread from the cpu core that each ran on, in a variable and then write that information to the output file. 

If you like, you may look at the code by doing:

```
vi hello_mpi-omp.c
```
To close the file from vi do `esc`, followed by `:q`. 

In real HPC applications, MPI tasks and OpenMP processes are used to organize the parallel tasks like solving matrix algebra or distributing data. MPI is used to share work between nodes, though it can also be used to share work between processors on the same node, while OpenMP designed exclusively to share work between the processors of a single node. Using both these programming models together helps programmers optimize parallel performance of their codes, and they are just two of the several parallel programming models available to coders. 

The output of hello_mpi_omp should look like this:

```
MPI 000 - OMP 000 - HWT 001 - Node frontier035
```

| MPI TaskID   | OpenMP process ID  | Hardware Thread ID | Node ID          |
| :--------    |:-----------------  | :----------------- | :--------------  |
| MPI 000      | OMP 000            | HWT 001            | Node frontier035 |

This means that MPI task 000 and OpenMP process 000 ran on hardware thread 001 on node 35. 

Remember, the example's srun was setup for 1 node (-N 1), 1 MPI task (-n 1), with one core for each MPI task (-c 1) and 1 OpenMP process (export OMP_NUM_THREADS=1) .

```
export OMP_NUM_THREADS=1

srun -N 1 -n 1 -c 1 ./hello_mpi_omp
```

To see if you got the same result from your job, do: 

```
ls
```

You will see something like this:

```
hello_mpi_omp    hello_mpi_omp.o    submit.sl
hello_mpi_omp.c  Makefile         *srun_YOUR-JOB-NAME-<your-job-ID-number>.out*
```
do 

```
vi   srun_YOUR-JOB-NAME-<your-job-ID-number>

```
where you replace `YOUR-JOB-NAME` and `your-job-ID-number` with the name and number from the file you listed.  

If you don't see output that looks like, 
```
MPI 000 - OMP 000 - HWT 001 - Node frontier035
```
retrace your steps or ask for help from the instructors. 


Multiple MPI Tasks Exercise 
-------------------
OK now let’s run with 7 MPI tasks. 

Here is what we have learned about srun so far: 

| Options| Meaning                      |
| :---   |:-----                        |
| N      | Number of Nodes              |
| n      | MPI tasks per node           |
| c      | number of CPU cores per task |

Starting with our previous srun line: 

```
srun -N 1 -n 1 -c 1 ./hello_mpi_omp
```
Which option would you set to `7` to get 7 MPI tasks per node? 

Open your submit.sl with `vi` or your favored text editor and make the change. 

Then submit your job with `sbatch submit.sl. 

When your job is done, open the output file (looks like a variation of srun_myjob-397453.out) with 'vi' or a text editor. Does it look like this? :

```
MPI 001 - OMP 000 - HWT 009 - Node frontier143
MPI 004 - OMP 000 - HWT 033 - Node frontier143
MPI 000 - OMP 000 - HWT 001 - Node frontier143
MPI 002 - OMP 000 - HWT 017 - Node frontier143
MPI 003 - OMP 000 - HWT 025 - Node frontier143
MPI 005 - OMP 000 - HWT 041 - Node frontier143
MPI 006 - OMP 000 - HWT 049 - Node frontier143

```
If so, you successfully ran 7 MPI tasks per node. 

Multiple OpenMP Processes Exercise 
------------------------------

Let's now try to run two OpenMP processes per MPI task. 

In your batch script, submit.sl, the line that controls the number of OpenMP processes is: 

```
export OMP_NUM_THREADS=1
```
In this case it is asking for one OpenMP process per task. 

Edit submit.sl to change that line so you have two OpenMP process per task. 

How many CPU cores are available to each of your MPI tasks? Do you think your cores will be able to handle the load with two processes each? 

Submit your job to find out. Look at the output file when you are done.

If you ran on Frontier, your output would look like this: 

```
WARNING: Requested total thread count and/or thread affinity may result in
oversubscription of available CPU resources!  Performance may be degraded.
Explicitly set OMP_WAIT_POLICY=PASSIVE or ACTIVE to suppress this message.
Set CRAY_OMP_CHECK_AFFINITY=TRUE to print detailed thread-affinity messages.
MPI 000 - OMP 000 - HWT 001 - Node frontier139
MPI 000 - OMP 001 - HWT 001 - Node frontier139
MPI 001 - OMP 000 - HWT 009 - Node frontier139
MPI 001 - OMP 001 - HWT 009 - Node frontier139
MPI 002 - OMP 000 - HWT 017 - Node frontier139
MPI 002 - OMP 001 - HWT 017 - Node frontier139
MPI 003 - OMP 000 - HWT 025 - Node frontier139
MPI 003 - OMP 001 - HWT 025 - Node frontier139
.  .  .    
```
The CPU's cores could easily handle two processes each, in fact, the Frontier cores have two hardware threads each, but the default slrum setting on Frontier is to only schedule one hardware thread per core. This allows each process to have all the resources of the core. So, in the way we have submitted the job, each hardware thread had to handle two processes. You can see this in the example output by the fact that every two OMP processes share one HWT. That is not an ideal situation because the thread would need to wait for one process to finish before it could start running the other. That situation is called oversubscription and the reason for the warnings in my example output. 

A better plan is to reserve a core for each process in the MPI task. What would you need to change about your current srun line, in submit.sl, to get a core reserved for each process in each MPI task? 

Remember: 
| Options| Meaning                      |
| :---   |:-----                        |
| N      | Number of Nodes              |
| n      | MPI tasks per node           |
| c      | number of CPU cores per task |


Make that change and submit the job again. Check your output to see if the warning is gone and if each OMP process has a unique HWT ID number associated with it in the output. 


Putting It All Together Exercise
--------------------------------

You can see that with just a few of the possible srun options, we have a lot of control at the runtime of our program over how its work is laid out on the node! 

For your final exercise in the challenge, see if you can setup and run the following job layouts for hello_mpi_omp without getting errors. Check your output.  

1.)  2 MPI tasks with 3 OpenMP processes each, on one node. (Your output file should have 6 lines) 
2.)  8 MPI tasks with 4 OpenMP processes each, on one node. (Your output file should have 32 lines) 

When you are done, copy the path to one of those output files to the google sheet to show that you have done the exercise.

In summary, we explored the Slurm options for sbatch that allow us to reserve compute nodes via the scheduler. We explored a few of the srun options that control how the parallel job launcher lays out work on the node.  

If you want to learn more, see the Frontier User Documentation's Slurm section, where there are many more examples: [https://docs.olcf.ornl.gov/systems/frontier_user_guide.html#slurm](https://docs.olcf.ornl.gov/systems/frontier_user_guide.html#slurm).
