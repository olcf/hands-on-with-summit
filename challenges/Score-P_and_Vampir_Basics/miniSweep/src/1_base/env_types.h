/*---------------------------------------------------------------------------*/
/*!
 * \file   env_types.h
 * \author Wayne Joubert
 * \date   Wed Jan 15 16:06:28 EST 2014
 * \brief  Data structure declarations relevant to parallel environment.
 * \note   Copyright (C) 2014 Oak Ridge National Laboratory, UT-Battelle, LLC.
 */
/*---------------------------------------------------------------------------*/

/*=============================================================================

This file has cross-cutting dpendencies across multiple parallel APIs
since these are tightly coupled to the data structure.

=============================================================================*/

#ifndef _env_types_h_
#define _env_types_h_

#ifdef USE_MPI
#include "mpi.h"
#endif

#ifdef USE_CUDA
#include "cuda.h"
#endif

#ifdef __cplusplus
extern "C"
{
#endif

/*===========================================================================*/
/*---Types---*/

#ifdef USE_MPI
typedef MPI_Comm    Comm_t;
typedef MPI_Request Request_t;
#else
typedef int Comm_t;
typedef int Request_t;
#endif

#ifdef USE_CUDA
typedef cudaStream_t Stream_t;
#else
typedef int Stream_t;
#endif

/*===========================================================================*/
/*---Struct containing environment information---*/

typedef struct
{
#ifdef USE_MPI
  int    nproc_x_;    /*---Number of procs along x axis---*/
  int    nproc_y_;    /*---Number of procs along y axis---*/
  int    tag_;        /*---Next free message tag---*/
  Comm_t active_comm_;
  Bool_t is_proc_active_;
#endif
#ifdef USE_CUDA
  Bool_t   is_using_device_;
  Stream_t stream_send_block_;
  Stream_t stream_recv_block_;
  Stream_t stream_kernel_faces_;
#endif
} Env;

/*===========================================================================*/

#ifdef __cplusplus
} /*---extern "C"---*/
#endif

#endif /*---_env_types_h_---*/

/*---------------------------------------------------------------------------*/
