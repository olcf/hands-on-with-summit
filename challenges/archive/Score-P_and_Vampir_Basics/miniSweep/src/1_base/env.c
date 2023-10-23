/*---------------------------------------------------------------------------*/
/*!
 * \file   env.c
 * \author Wayne Joubert
 * \date   Wed Jan 15 16:06:28 EST 2014
 * \brief  Environment settings specific to programming APIs.
 * \note   Copyright (C) 2014 Oak Ridge National Laboratory, UT-Battelle, LLC.
 */
/*---------------------------------------------------------------------------*/

#include <string.h>
#include "sys/time.h"

#include "arguments.h"
#include "env.h"

#ifdef __cplusplus
extern "C"
{
#endif

/*===========================================================================*/
/*---Null object---*/

Env Env_null(void)
{
  Env result;
  memset( (void*)&result, 0, sizeof(Env) );
  return result;
}

/*===========================================================================*/
/*---Initialize for execution---*/

void Env_initialize( Env *env, int argc, char** argv )
{
  Env_mpi_initialize_(  env, argc, argv );
  Env_cuda_initialize_( env, argc, argv );
}

/*===========================================================================*/
/*---Set values from args---*/

void Env_set_values( Env *env, Arguments* args )
{
  Env_mpi_set_values_(  env, args );
  Env_cuda_set_values_( env, args );
}

/*===========================================================================*/
/*---Finalize execution---*/

void Env_finalize( Env* env )
{
  Env_cuda_finalize_( env );
  Env_mpi_finalize_(  env );
}

/*===========================================================================*/
/*---Indicate whether to do output---*/

Bool_t Env_is_proc_master( Env* env )
{
  return ( Env_is_proc_active( env ) && Env_proc_this( env ) == 0 );
}

/*===========================================================================*/
/*---Timer utilities---*/

Timer Env_get_time( Env* env )
{
  struct timeval tv;
  int i = gettimeofday( &tv, NULL );
  Timer result = ( (Timer) tv.tv_sec +
                   (Timer) tv.tv_usec * 1.e-6 );
  return result;
}

/*---------------------------------------------------------------------------*/

Timer Env_get_synced_time( Env* env )
{
  Env_mpi_barrier( env );
  return Env_get_time( env );
}

/*===========================================================================*/

#ifdef __cplusplus
} /*---extern "C"---*/
#endif

/*---------------------------------------------------------------------------*/
