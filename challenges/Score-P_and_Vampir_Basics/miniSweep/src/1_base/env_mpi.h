/*---------------------------------------------------------------------------*/
/*!
 * \file   env_mpi.h
 * \author Wayne Joubert
 * \date   Wed Jan 15 16:06:28 EST 2014
 * \brief  Environment settings for MPI, header.
 * \note   Copyright (C) 2014 Oak Ridge National Laboratory, UT-Battelle, LLC.
 */
/*---------------------------------------------------------------------------*/

#ifndef _env_mpi_h_
#define _env_mpi_h_

#include <string.h>

#ifdef USE_MPI
#include "mpi.h"
#endif

#include "types.h"
#include "arguments.h"
#include "env_assert.h"

#ifdef __cplusplus
extern "C"
{
#endif

/*===========================================================================*/
/*---Initialize mpi---*/

void Env_mpi_nullify_values_( Env *env );

/*---------------------------------------------------------------------------*/

void Env_mpi_initialize_( Env *env, int argc, char** argv );

/*===========================================================================*/
/*---Determine whether set---*/

Bool_t Env_mpi_are_values_set_( const Env *env );

/*===========================================================================*/
/*---Finalize mpi---*/

void Env_mpi_finalize_values_( Env* env );

/*---------------------------------------------------------------------------*/

void Env_mpi_finalize_( Env* env );

/*===========================================================================*/
/*---Set values from args---*/

void Env_mpi_set_values_( Env *env, Arguments* args );

/*===========================================================================*/
/*---Get communicator being used---*/

Comm_t Env_mpi_active_comm_( const Env* env );

/*===========================================================================*/
/*---Number of procs---*/

int Env_nproc_x( const Env* env );

/*---------------------------------------------------------------------------*/

int Env_nproc_y( const Env* env );

/*---------------------------------------------------------------------------*/

int Env_nproc( const Env* env );

/*===========================================================================*/
/*---Is this proc within the subcommunicator of procs in use---*/

Bool_t Env_is_proc_active( const Env* env );

/*===========================================================================*/
/*---Tag manipulation---*/

int Env_tag( const Env* env );

/*---------------------------------------------------------------------------*/

void Env_increment_tag( Env* env, int value );

/*===========================================================================*/
/*---Proc number info---*/

int Env_proc( const Env* env, int proc_x, int proc_y );

/*---------------------------------------------------------------------------*/

int Env_proc_x( const Env* env, int proc );

/*---------------------------------------------------------------------------*/

int Env_proc_y( const Env* env, int proc );

/*===========================================================================*/
/*---Proc number info for this proc---*/

int Env_proc_this( const Env* env );

/*---------------------------------------------------------------------------*/

int Env_proc_x_this( const Env* env );

/*---------------------------------------------------------------------------*/

int Env_proc_y_this( const Env* env );

/*===========================================================================*/
/*---MPI functions: global MPI operations---*/

void Env_mpi_barrier( Env* env );

/*---------------------------------------------------------------------------*/

double Env_sum_d( Env* env, double value );

/*---------------------------------------------------------------------------*/

P Env_sum_P( Env* env, P value );

/*---------------------------------------------------------------------------*/

void Env_bcast_int( Env* env, int* data, int root );

/*---------------------------------------------------------------------------*/

void Env_bcast_string( Env* env, char* data, int len, int root );

/*===========================================================================*/
/*---MPI functions: point-to-point communication: synchronous---*/

void Env_send_i( Env* env, const int* data, size_t n, int proc, int tag );

/*---------------------------------------------------------------------------*/

void Env_recv_i( Env* env, int* data, size_t n, int proc, int tag );

/*---------------------------------------------------------------------------*/

void Env_send_P( Env* env, const P* data, size_t n, int proc, int tag );

/*---------------------------------------------------------------------------*/

void Env_recv_P( Env* env, P* data, size_t n, int proc, int tag );

/*===========================================================================*/
/*---MPI functions: point-to-point communication: asynchronous---*/

void Env_asend_P( Env* env, const P* data, size_t n, int proc, int tag,
                                                          Request_t* request );

/*---------------------------------------------------------------------------*/

void Env_arecv_P( Env* env, const P* data, size_t n, int proc, int tag,
                                                          Request_t* request );

/*---------------------------------------------------------------------------*/

void Env_wait( Env* env, Request_t* request );

/*===========================================================================*/

#ifdef __cplusplus
} /*---extern "C"---*/
#endif

#endif /*---_env_mpi_h_---*/

/*---------------------------------------------------------------------------*/
