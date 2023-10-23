#==============================================================================

cmake_minimum_required(VERSION 2.8)

project("minisweep")

SET(INCLUDE_DIRS
  ${CMAKE_SOURCE_DIR}/src/1_base
  ${CMAKE_SOURCE_DIR}/src/2_sweeper_base
  ${CMAKE_SOURCE_DIR}/src/3_sweeper
  ${CMAKE_SOURCE_DIR}/src/4_driver
  )

SET(SOURCES
  src/1_base/arguments.c
  src/1_base/env.c
  src/1_base/env_assert.c
  src/1_base/env_cuda.c
  src/1_base/env_mpi.c
  src/1_base/pointer.c
  src/2_sweeper_base/array_operations.c
  src/2_sweeper_base/dimensions.c
  src/3_sweeper/faces_kba.c
  src/3_sweeper/quantities.c
  src/3_sweeper/stepscheduler_kba.c
  src/3_sweeper/sweeper.c
  src/3_sweeper/sweeper_kernels.c
  src/4_driver/runner.c
  )

SET(CUDA_SOURCES)
FOREACH(FILE IN LISTS SOURCES)
  STRING(REPLACE ".c" ".cu" FILE2 ${FILE})
  SET(CUDA_SOURCES ${CUDA_SOURCES} ${FILE2})
ENDFOREACH()

#set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS} -Werror")

IF(USE_MPI)
  find_package(MPI REQUIRED)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DUSE_MPI")
ENDIF()

IF(USE_CUDA)
  find_package(CUDA REQUIRED)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DUSE_CUDA")
  STRING(REPLACE " " ";" CMAKE_C_FLAGS_ADJUSTED ${CMAKE_C_FLAGS})
  SET(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};${CMAKE_C_FLAGS_ADJUSTED}")
#  STRING(REGEX MATCH " -DNM_VALUE=[0-9]*" NM_VALUE_DEF_ " ${CMAKE_C_FLAGS} ")
#  STRING(REPLACE " " ";" NM_VALUE_DEF "${NM_VALUE_DEF_}")
#  SET(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS}${NM_VALUE_DEF};-DUSE_CUDA")
  CUDA_INCLUDE_DIRECTORIES(${INCLUDE_DIRS})
  CUDA_ADD_LIBRARY(sweeper STATIC ${CUDA_SOURCES})
  CUDA_ADD_EXECUTABLE(sweep src/4_driver/sweep.cu)
  TARGET_LINK_LIBRARIES(sweep sweeper)
  CUDA_ADD_EXECUTABLE(tester src/4_driver/tester.cu)
  TARGET_LINK_LIBRARIES(tester sweeper)
ELSE()
  INCLUDE_DIRECTORIES(${INCLUDE_DIRS})
  ADD_LIBRARY(sweeper STATIC ${SOURCES})
  ADD_EXECUTABLE(sweep src/4_driver/sweep.c)
  TARGET_LINK_LIBRARIES(sweep sweeper)
  ADD_EXECUTABLE(tester src/4_driver/tester.c)
  TARGET_LINK_LIBRARIES(tester sweeper)
ENDIF()

install(TARGETS sweep DESTINATION bin)
#install(TARGETS tester DESTINATION bin)

SET(CMAKE_SHARED_LIBRARY_LINK_C_FLAGS)

IF(CMAKE_C_COMPILER_ID STREQUAL "GNU")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -ansi -Wmissing-prototypes -Dinline=")
ENDIF()

IF(CMAKE_C_COMPILER_ID STREQUAL "Intel")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Dinline=__forceinline -D__restrict__=restrict -restrict")
ENDIF()

MESSAGE("Compiler ID: ${CMAKE_C_COMPILER_ID}")

# enable_testing()
# add_test(test1 tester)

#==============================================================================
