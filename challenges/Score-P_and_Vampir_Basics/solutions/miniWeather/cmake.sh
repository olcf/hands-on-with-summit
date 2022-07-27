#!/bin/bash

source ${MODULESHOME}/init/bash
module purge
module load DefApps gcc/9.1.0 cuda parallel-netcdf cmake
module unload darshan-runtime
module load scorep otf2 cubew

export TEST_MPI_COMMAND="jsrun -n 1 -c 1 -a 1 -g 1"

source cmake_clean.sh

export ASCENT_PARALLEL_NETCDF=/sw/ascent/spack-envs/base/opt/linux-rhel8-ppc64le/gcc-7.5.0/parallel-netcdf-1.12.2-dnfq2vfdlwarq5tif4iqjm5va75644cd

cmake -DCMAKE_CXX_COMPILER=scorep-mpicxx                                                     \
      -DCXXFLAGS="-O3 -std=c++11 -I${ASCENT_PARALLEL_NETCDF}/include"   \
      -DLDFLAGS="-L${ASCENT_PARALLEL_NETCDF}/lib -lpnetcdf"                        \
      -DOPENMP_FLAGS="-fopenmp"                                                       \
      -DNX=200                                                                        \
      -DNZ=100                                                                        \
      -DDATA_SPEC="DATA_SPEC_GRAVITY_WAVES"                                           \
      -DSIM_TIME=1000                                                                 \
      .
