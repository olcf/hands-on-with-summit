#include "stdio.h"
#include "mpi.h"

int main(int argc, char **argv)
{
    int rank, size;

    /*----------------*/
    /* Initialize MPI */
    /*----------------*/

     MPI_Init(&argc, &argv);

    /*------------------------------------------------------*/
    /* Get the number of ranks (size) from the Communicator */

    /*------------------------------------------------------*/

     MPI_Comm_size(MPI_COMM_WORLD, &size);


    /*------------------------------------------------------*/
    /* Get the Rank ID for each process                     */
    /* Fix the code here.                                   */
    /*------------------------------------------------------*/


    /*------------------------------------------------------*/
    /* Print hello from each rank                           */
    /*------------------------------------------------------*/

     printf("Hello from rank %d of %d total \n", rank, size);

    /*------------------------------------------------------*/
    /* Clean up and finalize the MPI environment            */
    /*------------------------------------------------------*/

     MPI_Finalize();

    return 0;
}

