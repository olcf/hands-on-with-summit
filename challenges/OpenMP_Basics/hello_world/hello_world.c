#define _GNU_SOURCE

#include <stdio.h>
#include <sched.h>
#include <omp.h>

int main(int argc, char *argv[]){

  int num_threads;
  int thread_id;
  int virtual_core;

  #pragma omp parallel default(none) shared(num_threads) private(thread_id, virtual_core)
  {
    num_threads = omp_get_num_threads();
    thread_id = omp_get_thread_num();
    virtual_core  = sched_getcpu();

    printf("OpenMP thread %03d of %03d ran on virtual core %03d\n", thread_id, num_threads, virtual_core);
  }

  return 0;
}
