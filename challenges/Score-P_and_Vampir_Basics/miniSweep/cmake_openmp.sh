#!/bin/bash -l
#------------------------------------------------------------------------------

# CLEANUP
rm -rf CMakeCache.txt
rm -rf CMakeFiles

# SOURCE AND INSTALL
if [ "$SOURCE" = "" ] ; then
  SOURCE=../minisweep
fi
if [ "$INSTALL" = "" ] ; then
  INSTALL=../install
fi

if [ "$BUILD" = "" ] ; then
  BUILD=Debug
  #BUILD=Release
fi

if [ "$NM_VALUE" = "" ] ; then
  NM_VALUE=4
fi

if [ "$PE_ENV" = INTEL ] ; then
  CC=icc
  OMP_ARGS="-qopenmp"
  OPT_ARGS="-ip -prec-div -O3 -align -ansi-alias -fargument-noalias -fno-alias -fargument-noalias"
else
  CC=mpicc
  OMP_ARGS="-fopenmp"
  OPT_ARGS="-O3 -fomit-frame-pointer -funroll-loops -finline-limit=10000000"
fi

#------------------------------------------------------------------------------

cmake \
  -DCMAKE_BUILD_TYPE:STRING="$BUILD" \
  -DCMAKE_INSTALL_PREFIX:PATH="$INSTALL" \
 \
  -DCMAKE_C_COMPILER:STRING="$CC" \
  -DCMAKE_C_FLAGS:STRING="-DNM_VALUE=$NM_VALUE -DUSE_OPENMP -DUSE_OPENMP_THREADS $OMP_ARGS -DUSE_MPI" \
  -DCMAKE_C_FLAGS_DEBUG:STRING="-g" \
  -DCMAKE_C_FLAGS_RELEASE:STRING="$OPT_ARGS" \
 \
 ..

#------------------------------------------------------------------------------
