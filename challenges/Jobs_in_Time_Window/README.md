# Inspecting Jobs

Developing large, parallel, scalable applications is arguably the most demanding effort that end-users of HPC systems like Frontier face. However, once an application is ready for production runs, a strong understanding of and familiarity with the user environment can be just as critical for a team to be productive.

The user environment includes interfaces to the batch scheduler, parallel job laucher, structure of available file systems, along with any user-configurable parts of the system. This challenge will rely on interaction with Frontier's batch scheduler, [SchedMD’s Slurm Workload Manager](https://slurm.schedmd.com/overview.html).

## Monitoring jobs with Slurm

As you may have already seen in [Basic_Workflow](../Basic_Workflow), Slurm provides the fundamental mechanisms to [submit batch jobs](https://docs.olcf.ornl.gov/systems/frontier_user_guide.html#batch-scripts) and [moderate submitted jobs](https://docs.olcf.ornl.gov/systems/frontier_user_guide.html#monitoring-and-modifying-batch-jobs) after they've been enqueued. 

We won't be submitting any new jobs here, but rather looking at others that have already been run and gathering information about them. To do this, we'll primarily use the `squeue` command. 

(See the `squeue` manual page by running `man` `squeue` for a full list of command options.)

## Let's try to answer...
1. How many jobs were completed on Frontier between 00:00 (midnight) on June 1, 2023 and 23:59 on June 15, 2023?

2. How many unique users did the jobs from question 1 belong to?

### BONUS!
3. Of the jobs found in question 1, what's the job ID of the longest running job?
    1. How long was it pending (pre-execution), and how long did it run (actual execution time)?
    2. When was it submitted?

