# hello_mpi.py
# Author: Michael A. Sandoval
from mpi4py import MPI

comm = MPI.COMM_WORLD      # Use the world communicator
mpi_rank = comm.Get_rank() # The process ID (integer 0-41 for a 42-process job)

print('Hello from MPI rank %s !' %(mpi_rank))
