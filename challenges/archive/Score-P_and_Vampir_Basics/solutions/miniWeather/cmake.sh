#!/bin/bash

source ${MODULESHOME}/init/bash
module purge
module load DefApps gcc/9.1.0 cuda parallel-netcdf cmake
module unload darshan-runtime
module load scorep otf2 cubew

export TEST_MPI_COMMAND="jsrun -n 1 -c 1 -a 1 -g 1"

source cmake_clean.sh


SCOREP_WRAPPER=off cmake -DCMAKE_CXX_COMPILER=scorep-mpicxx                                                     \
      -DCXXFLAGS="-O3 -std=c++11 -I${OLCF_PARALLEL_NETCDF}/include"   \
      -DLDFLAGS="-L${OLCF_PARALLEL_NETCDF}/lib -lpnetcdf"                        \
      -DOPENMP_FLAGS="-fopenmp"                                                       \
      -DNX=200                                                                        \
      -DNZ=100                                                                        \
      -DDATA_SPEC="DATA_SPEC_GRAVITY_WAVES"                                           \
      -DSIM_TIME=1000                                                                 \
      .
