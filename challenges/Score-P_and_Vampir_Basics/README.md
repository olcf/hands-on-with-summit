# Score-P and Vampir Basics

## What is Score-P and Vampir?

[Score-P](https://www.vi-hps.org/projects/score-p) is a profiling tool used to collect data on high-performance computing (HPC) applications. This data can include things like time and memory each function takes as well as how parallel processes are communicating with each other. The data collected from Score-P can then be analyzed with Vampir, which generates an interactive visualization that shows what functions in the code are actually doing. These two tools help to identify problematic areas of a program in order to make it more efficient. 

## Steps

There are 5 main steps when using Score-P:

1. Instrumenting
2. Profiling Run
3. Filtering
4. Tracing Run
5. Analyzing with Vampir

In this tutorial, we will walk through each one of these steps in detail with two example programs. 

## Step 1: Instrumenting

First off, to instrument Score-P you must add the `scorep` command as a prefix to the compiler. This can be done directly in the command line:

```
$ scorep gcc -c test.c
$ scorep gcc -o test test.o 
```

It can also be done in a Makefile. The `scorep` command can either be inserted right before the compiler variable or in its own variable as shown below:

```
## Makefile Example

CCOMP = gcc
CFLAGS = 
PREP = scorep

run: test.o
	$(PREP) $(CCOMP) $(CFLAGS) test.o -o test
test.o: test.c
	$(PREP) $(CCOMP) $(CFLAGS) -c test.c
clean:
	rm -rf test *.o
```

If using CMake files, however, it becomes a little more tricky. You would have to use the wrapper method to instrument Score-P:

```
## CMake Example

$ SCOREP_WRAPPER=off cmake .. \
	-DCMAKE_C_COMPILER=scorep--gcc \
	-DCMAKE_CXX_COMPIER=scorep--g++ \
	-DCMAKE_FORTRAN_COMPILER=scorep--ftn
```

We will go over this more in the miniWeather example program later on. 

### Practice with Monte Carlo Example

Let's try instrumenting Score-P in a Makefile! The Monte Carlo program was created as a demonstration of how MPI can be used to find the value of pi with the Monte Carlo method. For more information, see the [Monte Carlo Estimate PI](http://www.selkie.macalester.edu/csinparallel/modules/MPIProgramming/build/html/calculatePi/Pi.html) page. 

First, make sure you are in the correct directory:

```
$ cd ~/hands-on-with-summit/challenges/Score-P_and_Vampir_Basics/monte_carlo
```

Then, you will need to load the following modules:

```
$ module unload darshan-runtime
$ module load scorep gcc 
```

> NOTE: Loading the `scorep` module requires you to unload the `darshan-runtime` module first

Now, open the Makefile file with a text editor like vim:

```
$ vi Makefile
```

Add the `scorep` command in front of the `mpicc` compiler:

```
CCOMP = scorep mpicc
```

Then you can compile with the `make` command:

```
$ make
scorep mpicc  -c calcPiMPI.c
scorep mpicc  calcPiMPI.o -o run
```

You have officially instrumented Score-P on this program and are ready to do a profiling run!

## Step 2: Profiling Run

In order to perform a profiling run, you will need to set some Score-P environment variables. By default, SCOREP_ENABLE_PROFILING is set to true and SCOREP_ENABLE_TRACING is set to false, but it's good to get in the habit of setting both of these variables before every profiing and tracing run. To see more Score-P variables and their default values, you can always use the `scorep-info config-vars --full` command.

These variables are set using the `export` command as seen below:

```
$ export SCOREP_ENABLE_PROFILING=true
```
Running a program with profiling enabled will produce a new directory that contains the profiling data. We will see what we can do with this as we profile the Monte Carlo program.

### Practice with Monte Carlo Example

First set the appropriate environment varibles:

```
$ export SCOREP_ENABLE_PROFILING=true
$ export SCOREP_ENABLE_TRACING=false
```

Then you can submit the `submit.lsf` file with the `bsub` command to run the program.

```
$ bsub submit.lsf
```

To check the status of your job, use the `jobstat -u <username>` command. Once the job is complete, you should have a `monte_carlo.<job id>` file with the output of the program. You should also have a directory formatted `scorep-<yyyymmdd>-<time>-<unique id>`. This directory is named with the date and time it was created as well as a unique ID. Go to this directory using the `cd` command and list the contents with `ls`. You should see three files as shown below:

```
$ cd <scorep directory>
$ cd ls
MANIFEST.md  profile.cubex  scorep.cfg
```

To see profile data that Score-P collected, issue this command:

```
$ scorep-score -r profile.cubex
```

You should then see something like this:

```
Estimated aggregate size of event trace:                   2366 bytes
Estimated requirements for largest trace buffer (max_buf): 474 bytes
Estimated memory requirements (SCOREP_TOTAL_MEMORY):       4097kB
(hint: When tracing set SCOREP_TOTAL_MEMORY=4097kB to avoid intermediate flushes
 or reduce requirements using USR regions filters.)

flt     type max_buf[B] visits time[s] time[%] time/visit[us]  region
         ALL        473     60    0.91   100.0       15237.57  ALL
         MPI        360     40    0.91   100.0       22846.76  MPI
         COM         48     10    0.00     0.0          23.34  COM
      SCOREP         41      5    0.00     0.0          21.54  SCOREP
         USR         24      5    0.00     0.0           8.48  USR

         MPI        132     10    0.00     0.0          17.38  MPI_Reduce
         MPI         66      5    0.00     0.0           7.64  MPI_Barrier
         MPI         66      5    0.00     0.0          20.51  MPI_Bcast
      SCOREP         41      5    0.00     0.0          21.54  run
         MPI         24      5    0.00     0.0           1.61  MPI_Comm_rank
         MPI         24      5    0.00     0.0           4.99  MPI_Comm_size
         MPI         24      5    0.00     0.0          32.95  MPI_Finalize
         MPI         24      5    0.91    99.9      182671.63  MPI_Init
         COM         24      5    0.00     0.0          42.23  main
         COM         24      5    0.00     0.0           4.45  Get_input
         USR         24      5    0.00     0.0           8.48  Toss
```

The first line is the estimated size of the trace file, in this case 2,366 bytes. The second line estimates the memory space needed on one process for a trace (474 bytes) and the third line estimates the total memory per process required by Score-P (4,097 kilobytes). In the table, you can see all the regions and region types along with information like the amount of space and time those regions take as well as the number of visits to those regions in the program. 

With this example, we can see that MPI functions that up most of the time, specifically the ` MPI_Init` function. 

## Step 3: Filtering

Before performing a tracing run, we need to use the data from the profiling run to filter out the regions. This makes sure the size of the trace file isn't too large. MPI and OpenMP regions can't be filtered out, so it is recommeded to filter out USR regions that are short, frequently called, and overall not super important to measure. 

To filter the profile results, we first create a filter file in which you can specify what regions to exclude and/or include. `SCOREP_REGION_NAMES_BEGIN` and `SCOREP_REGION_NAMES_END` are used to define the filter block.

```
# Filter File Example
SCOREP_REGION_NAMES_BEGIN
	EXCUDE *
	INCLUDE region_one
		region_two
		region_three
SCOREP_REGION_NAMES_END
```

In the above example, the `*` wildcard is used to signify all the regions. Basically, we're telling Score-P to filter out all regions except for the three regions specified after `INCLUDE`. To see the effects of the filter, you run `scorep-score` again, but this time include the `-f <filter file name>` flag:

```
$ scorep-score -f test.filter  profile.cubex
```

Below is an example of the outcome of this command with a different program:

```
Estimated aggregate size of event trace:                   474MB
Estimated requirements for largest trace buffer (max_buf): 79MB
Estimated memory requirements (SCOREP_TOTAL_MEMORY):       93MB
(hint: When tracing set SCOREP_TOTAL_MEMORY=93MB to avoid intermediate flushes
 or reduce requirements using USR regions filters.)

flt     type max_buf[B]     visits time[s] time[%] time/visit[us]  region
 -       ALL 86,123,768 11,336,078   98.15   100.0           8.66  ALL
 -       OMP 78,915,165 10,219,230   88.52    90.2           8.66  OMP
 -       MPI  3,861,412    341,966    6.39     6.5          18.68  MPI
 -       USR  1,905,020    432,246    0.20     0.2           0.47  USR
 -       COM  1,484,730    342,630    3.04     3.1           8.87  COM
 -    SCOREP         41          6    0.00     0.0          50.49  SCOREP

 *       ALL 82,776,618 10,561,202   94.91    96.7           8.99  ALL-FLT
 -       OMP 78,915,165 10,219,230   88.52    90.2           8.66  OMP-FLT
 -       MPI  3,861,412    341,966    6.39     6.5          18.68  MPI-FLT
 +       FLT  3,389,750    774,876    3.24     3.3           4.19  FLT
 -    SCOREP         41          6    0.00     0.0          50.49  SCOREP-FLT
```

Under the `flt` column, the `-` symbol means "not filterd", the `+` symbol means "filtered", and the `*` symbol means "possibly affected by the filter". In the above example, it shows that a little less than 3.4 Megabytes were filtered out and will  be ignored during the trace.

Before submitting a job for a trace run, make sure you set the `SCOREP_FILTERING_FILE` enviroment variable to the filter file you created. For example:

```
$ export SCOREP_FILTERING_FILE=test.filter
```

### Practice with Monte Carlo Example

Now let's apply this to the Monte Carlo Example! First create a new file with a text editor like vim called `all.filter`:

```
$ vi all.filter
```

Since there is not a lot of USR or COM regions in the program to begin with, we will filter out all possible regions so that we are left with just MPI regions. However, if you want to see how some of the USR or COM functions run, feel free to go back and include them. To filter out all the USR and COM regions, add the following lines to the `all.filter` file:

```
SCOREP_REGION_NAMES_BEGIN
	EXCLUDE *
SCOREP_REGION_NAMES_END
```
Then issue the command:

```
$ scorep-score -r -f all.filter profile.cubex
```
And you should see somthing like this:

```
Estimated aggregate size of event trace:                   2006 bytes
Estimated requirements for largest trace buffer (max_buf): 402 bytes
Estimated memory requirements (SCOREP_TOTAL_MEMORY):       4097kB
(hint: When tracing set SCOREP_TOTAL_MEMORY=4097kB to avoid intermediate flushes
 or reduce requirements using USR regions filters.)

flt     type max_buf[B] visits time[s] time[%] time/visit[us]  region
 -       ALL        473     60    0.91   100.0       15237.57  ALL
 -       MPI        360     40    0.91   100.0       22846.76  MPI
 -       COM         48     10    0.00     0.0          23.34  COM
 -    SCOREP         41      5    0.00     0.0          21.54  SCOREP
 -       USR         24      5    0.00     0.0           8.48  USR

 *       ALL        401     45    0.91   100.0       20310.62  ALL-FLT
 -       MPI        360     40    0.91   100.0       22846.76  MPI-FLT
 +       FLT         72     15    0.00     0.0          18.39  FLT
 -    SCOREP         41      5    0.00     0.0          21.54  SCOREP-FLT

 -       MPI        132     10    0.00     0.0          17.38  MPI_Reduce
 -       MPI         66      5    0.00     0.0           7.64  MPI_Barrier
 -       MPI         66      5    0.00     0.0          20.51  MPI_Bcast
 -    SCOREP         41      5    0.00     0.0          21.54  run
 -       MPI         24      5    0.00     0.0           1.61  MPI_Comm_rank
 -       MPI         24      5    0.00     0.0           4.99  MPI_Comm_size
 -       MPI         24      5    0.00     0.0          32.95  MPI_Finalize
 -       MPI         24      5    0.91    99.9      182671.63  MPI_Init
 +       COM         24      5    0.00     0.0          42.23  main
 +       COM         24      5    0.00     0.0           4.45  Get_input
 +       USR         24      5    0.00     0.0           8.48  Toss
```

So now the trace we will run will have only MPI regions since we filtered out both USR and COM regions. 

Next, assign the `all.filter` file to the `SCOREP_FILTERING_FILE` variable:

```
$ export SCOREP_FILTERING_FILE=all.filter
```

## Step 4: Tracing Run

After filtering out unwanted data, we can finally perform a tracing run. A trace will give us a more detailed look at what is happening inside the program. A tracing run is very similar to a profiling run but needs different Score-P environemt variable values.

### Practice with Monte Carlo Example

First, in addition to the `SCOREP_FILTERING_FILE` variable we set in the last step, there are other Score-P environemnt variables to set before running a trace measurement:

```
$ export SCOREP_ENABLE_PROFILING=false
$ export SCOREP_ENABLE_TRACING=true
$ export SCOREP_TOTAL_MEMORY=4097kB
```

Like the profiling run, the variables `SCOREP_ENABLE_PROFILING`` and `SCOREP_ENABLE_TRACING` must be set to specify what action will take place during the run. It's also good to set `SCOREP_TOTAL_MEMORY` to the recommended amount in order to reduce the amount of memory required and to avoid flushes.

Once we have these variables set, get out of the scorep directory and submit the job!

```
$ cd ..
$ bsub submit.lsf
```

When the job is complete, you should see another directory named `scorep-<yyyymmdd>-<time>-<unique id>. It should contain a `traces.otf2` file that we can use to view the trace with Vampir in the next step.

```
$ cd <new scorep directory>
$ ls
MANIFEST.md  scorep.cfg  traces  traces.def  traces.otf2  traces.vsettings  
```

## Step 5: Analyzing with Vampir

Now for the cool part. The `otf2` trace file generated by Score-P can be loaded on Vampir. There are a lot of graphs and timelines that can be used to analyze a program, but one major one is the Master Timeline. This feature shows the order of the functions called in each process as well as all the messages relayed between them. The black diamonds throughout the timeline signify message bursts (the darker and bigger the diamond, the more message activity there is) and the yellow arrows signify I/O events. You can zoom in on the timeline to get a closer look at these messages.

There are also function summaries that can tell you how much times is spent in each function across all processes as well as process summaries that focus on one process' function call stack.  

### Practice with Monte Carlo Example

Now it's your turn!

In order to load Vampir on Ascent, you must have X forwarding. This process differs between Mac and Windows users, so follow the appropriate directions for what you are working with.

**Mac Users**
To log in with X forwarding, logout of Ascent, and then ssh back in with the -X flag:

```
$ ssh -X <username>@login1.ascent.olcf.ornl.gov
```

Another thing you will need to have is the XQuartz app. This can be accessed by going to ORNL Self Service in your launchpad (can also use search bar at the top to search for it). In ORNL Self Service, scroll down until you find XQuartz and click the Install button. 

**Windows Users**

Please go to the following links to set up Xming and Putty:

[Installing/Configuring PuTTy and Xming](http://www.geo.mtu.edu/geoschem/docs/putty_install.html)
[PuTTy Setup with SSH and X11](http://ocean.stanford.edu/courses/ESS141/PuTTY/)

While following the second link's instructions, use `login1.ascent.olcf.ornl.gov` hostname instead of "ocean.stanford.edu" and use your own username instead of "gp235". 

Once you have the necessary software and are logged in with X forwarding, you can load the Vampir module to your environment:

```
$ module load vampir
```
You should see a message like this:

```
$ Run VampirServer via
  $ vampirserver start -- -P <projid> [-q <queue>] [-w <walltime>]
and follow the instructions provided by this script.
Run the Vampir GUI remotely via X-forwarding
  $ vampir &
```

We will be using the `vampir &` command to lauch Vampir remotely since our trace file isn't very large. For large trace files (over 1 GB), see the [VampirServer Documentation](https://docs.olcf.ornl.gov/software/profiling/Vampir.html#vampir-using-vampirserver).

To lauch Vampir:

```
$ vampir &
```

>NOTE: If you get this message:

	```
	$ [Warning] could not connect to display localhost:12.0
	[Fatal] This application failed to start because no Qt platform plugin could be initialized. Reinstalling the application may fix this problem.

	Available platform plugins are: xcb.
	```

	You either need to logout and log back in or there is something wrong with the XQuartz/Xming application. You might need to restart your computer to get the application going. 

This should open a Vampir window that looks like this:

enter image.

To open the `otf2` file we created, click on "Local File" on the left (if the window shows a Recent Files page, just click "Open Other"). Then enter in the path to the file, in this case it would be `ccsopen/home/<username>/hands-on-with-summit/challenges/Score-P_and_Vampir_Basics/<2nd scorep directory>/traces.otf2`. After the file loads, you should see a Vampir GUI like this:

enter image.
  
The master timeline is the big, crowded-looking frame on the left. This shows how each process runs over time and which functions are called according to the color-coded function group (the key for the colors is shown in the bottom right). The frame in the top-right corner is the function summary, which breaks down the accumulated time spent in each function summed up across all the processes. Then the middle frame on the right is the context view that gives details about the trace and the bottom-right frame provides a key for what colors represent what function groups. At the very very top is the trace timeline, which gives an general timeline of the trace.  

Feel free to play around with all the displays available. Try zooming in where there are any message bursts (black diamonds) to see what is going on. Clicking on anything in the master timeline will give you information on the feature in the context view frame. You might also want to use the the top-left bar to add new frames to the overall display. For example, this button (enter image) will open the process summary.

Now you have successsfully profiled a program with Score-P as well as opened and used Vampir! 

For more information on everything Vampir can do, see this [Vampir Visualizations Guide](https://vampir.eu/tutorial/manual/performance_data_visualization#sec-commmatrix) page. 

## MiniWeather Example Program

MiniWeather is a mini application used to simulate weather-like flows. For our purposes, we will be more focused on profiling the program instead of the output, but feel free to dive more into miniWeather's specifics at this [MiniWeather](https://github.com/mrnorman/miniWeather) Github page.

The MiniWeather app uses CMake to compile and build the code, which requires a different approach when instrumenting Score-P. This makes it more complicated, but let's walk through it.

Before starting, make sure you are in the right directory:

```
$ cd ~/hands-on-with-summit/challenges/Score-P_and_Vampir_Basics/miniWeather
``` 

## Challenges

## Further Resources

[Score-P](https://scorepci.pages.jsc.fz-juelich.de/scorep-pipelines/docs/scorep-6.0/html/index.html)

[Vampir Visualizations Guide](https://vampir.eu/tutorial/manual/performance_data_visualization#sec-commmatrix)

[Score-P OLCF Documentation](https://docs.olcf.ornl.gov/software/profiling/Scorep.html)

[Vampir OLCF Documentation](https://docs.olcf.ornl.gov/software/profiling/Vampir.html)

[OLCF Training Archives.](https://docs.olcf.ornl.gov/training/training_archive.html) There are two Score-P presentations called "Intro to Score-P" and "Advanced Score-P" that are listed. 

[Monte Carlo Code Explanation](http://www.selkie.macalester.edu/csinparallel/modules/MPIProgramming/build/html/calculatePi/Pi.html)

[MiniWeather Full Code and Explanation](https://github.com/mrnorman/miniWeather#running-the-code)


