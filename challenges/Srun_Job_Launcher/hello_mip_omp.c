/* -------------------------------------------------------------
MPI + OpenMP Hello, World program to help understand process
and thread mapping to physical CPU cores and hardware threads
------------------------------------------------------------- */
#define _GNU_SOURCE

#include <stdio.h>
#include <mpi.h>
#include <sched.h>
#include <omp.h>

int main(int argc, char *argv[]){

        MPI_Init(&argc, &argv);

        int size;
        MPI_Comm_size(MPI_COMM_WORLD, &size);

        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    char name[MPI_MAX_PROCESSOR_NAME];
    int resultlength;
    MPI_Get_processor_name(name, &resultlength);

    int hwthread;
    int thread_id = 0;

    #pragma omp parallel default(shared) private(hwthread, thread_id)
    {
        thread_id = omp_get_thread_num();
        hwthread  = sched_getcpu();

        printf("MPI %03d - OMP %03d - HWT %03d - Node %s\n", rank, thread_id, hwthread, name);
    }

        MPI_Finalize();
