/*---------------------------------------------------------------------------*/
/*!
 * \file   env_mpi.c
 * \author Wayne Joubert
 * \date   Wed Jan 15 16:06:28 EST 2014
 * \brief  Environment settings for MPI.
 * \note   Copyright (C) 2014 Oak Ridge National Laboratory, UT-Battelle, LLC.
 */
/*---------------------------------------------------------------------------*/

#include <string.h>

#ifdef USE_MPI
#include "mpi.h"
#endif

#include "types.h"
#include "env_types.h"
#include "arguments.h"
#include "env_mpi.h"

#ifdef __cplusplus
extern "C"
{
#endif

/*===========================================================================*/
/*---Initialize mpi---*/

void Env_mpi_nullify_values_( Env *env )
{
#ifdef USE_MPI
  /*---Initialize MPI-related variables in env struct to null---*/
  env->nproc_x_ = 0;
  env->nproc_y_ = 0;
  env->tag_ = 0;
  env->active_comm_ = 0;
  env->is_proc_active_ = 0;
#endif
}

/*---------------------------------------------------------------------------*/

void Env_mpi_initialize_( Env *env, int argc, char** argv )
{
#ifdef USE_MPI
  /*---Initialize for MPI execution---*/
  const int mpi_code = MPI_Init( &argc, &argv );
  Assert( mpi_code == MPI_SUCCESS );
#endif
  Env_mpi_nullify_values_( env );
}

/*===========================================================================*/
/*---Determine whether set---*/

Bool_t Env_mpi_are_values_set_( const Env *env )
{
  /*---NOTE: MPI can be initialized but the values not yet set---*/
  Bool_t result = Bool_true;
#ifdef USE_MPI
  result = env->nproc_x_ > 0 ? Bool_true : Bool_false;
#endif
  return result;
}

/*===========================================================================*/
/*---Finalize mpi---*/

void Env_mpi_finalize_values_( Env* env )
{
#ifdef USE_MPI
  if( Env_mpi_are_values_set_( env ) )
  {
    if( env->active_comm_ != MPI_COMM_WORLD )
    {
      const int mpi_code = MPI_Comm_free( &env->active_comm_ );
      Assert( mpi_code == MPI_SUCCESS );
    }
  }
#endif
    /*---Restore types to unset null state---*/
    Env_mpi_nullify_values_( env );
}

/*---------------------------------------------------------------------------*/

void Env_mpi_finalize_( Env* env )
{
  Env_mpi_finalize_values_( env );
#ifdef USE_MPI
  const int mpi_code = MPI_Finalize();
  Assert( mpi_code == MPI_SUCCESS );
#endif
}

/*===========================================================================*/
/*---Set values from args---*/

void Env_mpi_set_values_( Env *env, Arguments* args )
{
  /*---The purpose of this function is to separate the initialization
       of MPI from the initialization of the struct, so the struct can
       be reinitialized if needed---*/
  if( Env_mpi_are_values_set_( env ) )
  {
    Env_mpi_finalize_values_( env );
  }

#ifdef USE_MPI
  int mpi_code = 0;
  if( mpi_code ) {} /*---Remove unused var warning---*/

  env->nproc_x_ = Arguments_consume_int_or_default( args, "--nproc_x", 1 );
  env->nproc_y_ = Arguments_consume_int_or_default( args, "--nproc_y", 1 );
  Insist( env->nproc_x_ > 0 ? "Invalid nproc_x supplied." : 0 );
  Insist( env->nproc_y_ > 0 ? "Invalid nproc_y supplied." : 0 );

  const int nproc_requested = env->nproc_x_ * env->nproc_y_;
  int nproc_world = 0;
  mpi_code = MPI_Comm_size( MPI_COMM_WORLD, &nproc_world );
  Assert( mpi_code == MPI_SUCCESS );
  Insist( nproc_requested <= nproc_world ?
                                      "Not enough processors available." : 0 );

  int rank = 0;
  mpi_code = MPI_Comm_rank( MPI_COMM_WORLD, &rank );
  Assert( mpi_code == MPI_SUCCESS );

  /*---Set up a communicator for execution that may be smaller
       than the total number of ranks available, with consistent proc
       numbers---*/
  env->is_proc_active_ = rank < nproc_requested ? Bool_true : Bool_false;
  mpi_code = MPI_Comm_split( MPI_COMM_WORLD, env->is_proc_active_,
                                                   rank, &env->active_comm_ );
  Assert( mpi_code == MPI_SUCCESS );

  env->tag_ = 0;
#endif
}

/*===========================================================================*/
/*---Get communicator being used---*/

Comm_t Env_mpi_active_comm_( const Env* env )
{
  Assert( Env_mpi_are_values_set_( env ) );
#ifdef USE_MPI
  return env->active_comm_;
#else
  return 0;
#endif
}

/*===========================================================================*/
/*---Number of procs---*/

int Env_nproc_x( const Env* env )
{
  Assert( Env_mpi_are_values_set_( env ) );
  int result = 1;
#ifdef USE_MPI
  result = env->nproc_x_;
#endif
  Assert( result > 0 );
  return result;
}

/*---------------------------------------------------------------------------*/

int Env_nproc_y( const Env* env )
{
  Assert( Env_mpi_are_values_set_( env ) );
  int result = 1;
#ifdef USE_MPI
  result = env->nproc_y_;
#endif
  Assert( result > 0 );
  return result;
}

/*---------------------------------------------------------------------------*/

int Env_nproc( const Env* env )
{
  Assert( Env_mpi_are_values_set_( env ) );
  return Env_nproc_x( env ) * Env_nproc_y( env );
}

/*===========================================================================*/
/*---Is this proc within the subcommunicator of procs in use---*/

Bool_t Env_is_proc_active( const Env* env )
{
  Assert( Env_mpi_are_values_set_( env ) );
  Bool_t result = Bool_true;
#ifdef USE_MPI
  result = env->is_proc_active_ ? Bool_true : Bool_false;
#endif
  Assert( result==Bool_true || result==Bool_false );
  return result;
}

/*===========================================================================*/
/*---Tag manipulation---*/

int Env_tag( const Env* env )
{
  Assert( Env_mpi_are_values_set_( env ) );
  int result = 0;
#ifdef USE_MPI
  result = env->tag_;
#endif
  Assert( result >= 0 );
  return result;
}

/*---------------------------------------------------------------------------*/

void Env_increment_tag( Env* env, int value )
{
  Assert( Env_mpi_are_values_set_( env ) );
  Assert( value > 0 );
#ifdef USE_MPI
  env->tag_ += value;
#endif
}

/*===========================================================================*/
/*---Proc number info---*/

int Env_proc( const Env* env, int proc_x, int proc_y )
{
  Assert( Env_mpi_are_values_set_( env ) );
  Assert( proc_x >= 0 && proc_x < Env_nproc_x( env ) );
  Assert( proc_y >= 0 && proc_y < Env_nproc_y( env ) );
  int result = proc_x + Env_nproc_x( env ) * proc_y;
  Assert( result >= 0 && result < Env_nproc_x( env ) * Env_nproc_y( env ) );
  return result;
}

/*---------------------------------------------------------------------------*/

int Env_proc_x( const Env* env, int proc )
{
  Assert( Env_mpi_are_values_set_( env ) );
  Assert( proc >= 0 && proc < Env_nproc( env ) );
  int result = proc % Env_nproc_x( env );
  Assert( result >= 0 && result < Env_nproc_x( env ) );
  return result;
}

/*---------------------------------------------------------------------------*/

int Env_proc_y( const Env* env, int proc )
{
  Assert( Env_mpi_are_values_set_( env ) );
  Assert( proc >= 0 && proc < Env_nproc( env ) );
  int result = proc / Env_nproc_x( env );
  Assert( result >= 0 && result < Env_nproc_y( env ) );
  return result;
}

/*===========================================================================*/
/*---Proc number info for this proc---*/

int Env_proc_this( const Env* env )
{
  Assert( Env_mpi_are_values_set_( env ) );
  int result = 0;
#ifdef USE_MPI
  const int mpi_code = MPI_Comm_rank( Env_mpi_active_comm_( env ), &result );
  Assert( mpi_code == MPI_SUCCESS );
#endif
  Assert( result >= 0 && result < Env_nproc( env ) );
  return result;
}

/*---------------------------------------------------------------------------*/

int Env_proc_x_this( const Env* env )
{
  Assert( Env_mpi_are_values_set_( env ) );
  return Env_proc_x( env, Env_proc_this( env ) );
}

/*---------------------------------------------------------------------------*/

int Env_proc_y_this( const Env* env )
{
  Assert( Env_mpi_are_values_set_( env ) );
  return Env_proc_y( env, Env_proc_this( env ) );
}

/*===========================================================================*/
/*---MPI functions: global MPI operations---*/

void Env_mpi_barrier( Env* env )
{
  Assert( Env_mpi_are_values_set_( env ) );
#ifdef USE_MPI
  const int mpi_code = MPI_Barrier( Env_mpi_active_comm_( env ) );
  Assert( mpi_code == MPI_SUCCESS );
#endif
}

/*---------------------------------------------------------------------------*/

double Env_sum_d( Env* env, double value )
{
  Assert( Env_mpi_are_values_set_( env ) );
  double result = 0;
#ifdef USE_MPI
  const int mpi_code = MPI_Allreduce( &value, &result, 1, MPI_DOUBLE, MPI_SUM,
                                                Env_mpi_active_comm_( env ) );
  Assert( mpi_code == MPI_SUCCESS );
#else
  result = value;
#endif
  return result;
}

/*---------------------------------------------------------------------------*/

P Env_sum_P( Env* env, P value )
{
  Assert( Env_mpi_are_values_set_( env ) );
  Static_Assert( P_IS_DOUBLE );
  return Env_sum_d( env, value );
}

/*---------------------------------------------------------------------------*/

void Env_bcast_int( Env* env, int* data, int root )
{
  Assert( Env_mpi_are_values_set_( env ) );
#ifdef USE_MPI
  const int mpi_code = MPI_Bcast( data, 1, MPI_INT, root,
                                                Env_mpi_active_comm_( env ) );
  Assert( mpi_code == MPI_SUCCESS );
#endif
}

/*---------------------------------------------------------------------------*/

void Env_bcast_string( Env* env, char* data, int len, int root )
{
  Assert( Env_mpi_are_values_set_( env ) );
#ifdef USE_MPI
  const int mpi_code = MPI_Bcast( data, len, MPI_CHAR, root,
                                                Env_mpi_active_comm_( env ) );
  Assert( mpi_code == MPI_SUCCESS );
#endif
}

/*===========================================================================*/
/*---MPI functions: point-to-point communication: synchronous---*/

void Env_send_i( Env* env, const int* data, size_t n, int proc, int tag )
{
  Assert( Env_mpi_are_values_set_( env ) );
  Assert( data != NULL );
  Assert( n+1 >= 1 );
  Assert( proc>=0 && proc<Env_nproc( env ) );
  Assert( tag>=0 );

#ifdef USE_MPI
  const int mpi_code = MPI_Send( (void*)data, n, MPI_INT, proc, tag,
                                                Env_mpi_active_comm_( env ) );
  Assert( mpi_code == MPI_SUCCESS );
#endif
}

/*---------------------------------------------------------------------------*/

void Env_recv_i( Env* env, int* data, size_t n, int proc, int tag )
{
  Assert( Env_mpi_are_values_set_( env ) );
  Assert( data != NULL );
  Assert( n+1 >= 1 );
  Assert( proc>=0 && proc<Env_nproc( env ) );
  Assert( tag>=0 );

#ifdef USE_MPI
  MPI_Status status;
  const int mpi_code = MPI_Recv( (void*)data, n, MPI_INT, proc, tag,
                                       Env_mpi_active_comm_( env ), &status );
  Assert( mpi_code == MPI_SUCCESS );
#endif
}

/*---------------------------------------------------------------------------*/

void Env_send_P( Env* env, const P* data, size_t n, int proc, int tag )
{
  Assert( Env_mpi_are_values_set_( env ) );
  Static_Assert( P_IS_DOUBLE );
  Assert( data != NULL );
  Assert( n+1 >= 1 );
  Assert( proc>=0 && proc<Env_nproc( env ) );
  Assert( tag>=0 );

#ifdef USE_MPI
  const int mpi_code = MPI_Send( (void*)data, n, MPI_DOUBLE, proc, tag,
                                                Env_mpi_active_comm_( env ) );
  Assert( mpi_code == MPI_SUCCESS );
#endif
}

/*---------------------------------------------------------------------------*/

void Env_recv_P( Env* env, P* data, size_t n, int proc, int tag )
{
  Assert( Env_mpi_are_values_set_( env ) );
  Static_Assert( P_IS_DOUBLE );
  Assert( data != NULL );
  Assert( n+1 >= 1 );
  Assert( proc>=0 && proc<Env_nproc( env ) );
  Assert( tag>=0 );

#ifdef USE_MPI
  MPI_Status status;
  const int mpi_code = MPI_Recv( (void*)data, n, MPI_DOUBLE, proc, tag,
                                       Env_mpi_active_comm_( env ), &status );
  Assert( mpi_code == MPI_SUCCESS );
#endif
}

/*===========================================================================*/
/*---MPI functions: point-to-point communication: asynchronous---*/

void Env_asend_P( Env* env, const P* data, size_t n, int proc, int tag,
                                                          Request_t* request )
{
  Assert( Env_mpi_are_values_set_( env ) );
  Static_Assert( P_IS_DOUBLE );
  Assert( data != NULL );
  Assert( n+1 >= 1 );
  Assert( proc>=0 && proc<Env_nproc( env ) );
  Assert( tag>=0 );
  Assert( request != NULL );

#ifdef USE_MPI
  const int mpi_code = MPI_Isend( (void*)data, n, MPI_DOUBLE, proc, tag,
                                       Env_mpi_active_comm_( env ), request );
  Assert( mpi_code == MPI_SUCCESS );
#endif
}

/*---------------------------------------------------------------------------*/

void Env_arecv_P( Env* env, const P* data, size_t n, int proc, int tag,
                                                          Request_t* request )
{
  Assert( Env_mpi_are_values_set_( env ) );
  Static_Assert( P_IS_DOUBLE );
  Assert( data != NULL );
  Assert( n+1 >= 1 );
  Assert( proc>=0 && proc<Env_nproc( env ) );
  Assert( tag>=0 );
  Assert( request != NULL );

#ifdef USE_MPI
  const int mpi_code = MPI_Irecv( (void*)data, n, MPI_DOUBLE, proc, tag,
                                       Env_mpi_active_comm_( env ), request );
  Assert( mpi_code == MPI_SUCCESS );
#endif
}

/*---------------------------------------------------------------------------*/

void Env_wait( Env* env, Request_t* request )
{
#ifdef USE_MPI
  MPI_Status status;
  const int mpi_code = MPI_Waitall( 1, request, &status );
  Assert( mpi_code == MPI_SUCCESS );
#endif
}

/*===========================================================================*/

#ifdef __cplusplus
} /*---extern "C"---*/
#endif

/*---------------------------------------------------------------------------*/
