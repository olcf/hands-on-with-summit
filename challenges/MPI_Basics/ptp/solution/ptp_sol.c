#include <stdio.h>
#include <string.h>
#include "mpi.h"

int main(int argc, char ** argv) 
{
  int rank, ntag = 100;
  char message[14];
  MPI_Status status;

  /*----------------*/
  /* Initialize MPI */
  /*----------------*/

  MPI_Init(&argc, &argv);

  /*------------------------------------------------------*/
  /* Get my rank in the MPI_COMM_WORLD communicator group */
  /*------------------------------------------------------*/

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  /*---------------------------------------*/
  /* Process 0 sends a message to process 1*/
  /*---------------------------------------*/
	
  if (rank == 0) {
    strcpy(message, "hello!");
    MPI_Send(&message, 6, MPI_CHAR, 1, ntag, MPI_COMM_WORLD);
  }

  /*----------------------------------------------*/
  /* Process 1 receives a message from process 0  */
  /* and outputs the result                       */
  /* Use the send function and the definition     */
  /* of MPI_Recv to fill in A,B and C below       */ 
  /*----------------------------------------------*/

  if (rank == 1 ) {
    MPI_Recv(&message, 6, MPI_CHAR, 0, ntag, MPI_COMM_WORLD, &status);
    printf("Process %d : %s\n", rank, message);
  }

  /*--------------*/
  /* Finalize MPI */
  /*--------------*/

  MPI_Finalize();

}
