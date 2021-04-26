#include "stdio.h"
#include "mpi.h"

int main(int argc, char **argv)
{
    int rank, size;

    // Initialize MPI (Put the MPI initialization functions here)

     MPI_Init(&argc, &argv);  

    //Get the number of ranks (size) from the Communicator (Put the MPI size function here)
     
     MPI_Comm_size(MPI_COMM_WORLD, &size);     
    //Get the rank for each process from the communicator. (Put the MPI rank function here)
     
     MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    printf("Hello from rank %d of %d total \n", rank, size);
      

    //Clean up and finalize the MPI environment (Put the MPI finalization function here)
     MPI_Finalize();

    return 0;
}
