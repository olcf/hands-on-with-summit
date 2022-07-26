/*---------------------------------------------------------------------------*/
/*!
 * \file   env_cuda.c
 * \author Wayne Joubert
 * \date   Tue Apr 22 17:03:08 EDT 2014
 * \brief  Environment settings for cuda.
 * \note   Copyright (C) 2014 Oak Ridge National Laboratory, UT-Battelle, LLC.
 */
/*---------------------------------------------------------------------------*/

#include <stddef.h>
#include <stdlib.h>

#include "types.h"
#include "env_types.h"
#include "env_assert.h"
#include "arguments.h"
#include "env_cuda.h"

#ifdef __cplusplus
extern "C"
{
#endif


/*===========================================================================*/
/*---Error handling---*/

Bool_t Env_cuda_last_call_succeeded()
{
  Bool_t result = Bool_true;

#ifdef USE_CUDA
  /*---NOTE: this read of the last error is a destructive read---*/
  cudaError_t error = cudaGetLastError();

  if ( error != cudaSuccess )
  {
      result = Bool_false;
      printf( "CUDA error detected: %s\n", cudaGetErrorString( error ) );
  }
#endif

  return result;
}

/*===========================================================================*/
/*---Initialize CUDA---*/

void Env_cuda_initialize_( Env *env, int argc, char** argv )
{
#ifdef USE_CUDA
  cudaStreamCreate( & env->stream_send_block_ );
  Assert( Env_cuda_last_call_succeeded() );

  cudaStreamCreate( & env->stream_recv_block_ );
  Assert( Env_cuda_last_call_succeeded() );

  cudaStreamCreate( & env->stream_kernel_faces_ );
  Assert( Env_cuda_last_call_succeeded() );
#endif
}

/*===========================================================================*/
/*---Finalize CUDA---*/

void Env_cuda_finalize_( Env* env )
{
#ifdef USE_CUDA
  cudaStreamDestroy( env->stream_send_block_ );
  Assert( Env_cuda_last_call_succeeded() );

  cudaStreamDestroy( env->stream_recv_block_ );
  Assert( Env_cuda_last_call_succeeded() );

  cudaStreamDestroy( env->stream_kernel_faces_ );
  Assert( Env_cuda_last_call_succeeded() );
#endif
}

/*===========================================================================*/
/*---Set values from args---*/

void Env_cuda_set_values_( Env *env, Arguments* args )
{
#ifdef USE_CUDA
  env->is_using_device_ = Arguments_consume_int_or_default( args,
                                             "--is_using_device", Bool_false );
  Insist( env->is_using_device_ == 0 ||
          env->is_using_device_ == 1 ? "Invalid is_using_device value." : 0 );
#endif
}

/*===========================================================================*/
/*---Determine whether using device---*/

Bool_t Env_cuda_is_using_device( const Env* const env )
{
#ifdef USE_CUDA
  return env->is_using_device_;
#else
  return Bool_false;
#endif
}

/*===========================================================================*/
/*---Memory management, for CUDA and all platforms ex. MIC---*/

#ifndef __MIC__

int* malloc_host_int( size_t n )
{
  Assert( n+1 >= 1 );
  int* result = (int*)malloc( n * sizeof(int) );
  Assert( result );
  return result;
}

/*---------------------------------------------------------------------------*/

P* malloc_host_P( size_t n )
{
  Assert( n+1 >= 1 );
  P* result = (P*)malloc( n * sizeof(P) );
  Assert( result );
  return result;
}

/*---------------------------------------------------------------------------*/

P* malloc_host_pinned_P( size_t n )
{
  Assert( n+1 >= 1 );

  P* result = NULL;

#ifdef USE_CUDA
  cudaMallocHost( &result, n==0 ? ((size_t)1) : n*sizeof(P) );
  Assert( Env_cuda_last_call_succeeded() );
#else
  result = (P*)malloc( n * sizeof(P) );
#endif
  Assert( result );

  return result;
}

/*---------------------------------------------------------------------------*/

P* malloc_device_P( size_t n )
{
  Assert( n+1 >= 1 );

  P* result = NULL;

#ifdef USE_CUDA
  cudaMalloc( &result, n==0 ? ((size_t)1) : n*sizeof(P) );
  Assert( Env_cuda_last_call_succeeded() );
  Assert( result );
#endif

  return result;
}

/*---------------------------------------------------------------------------*/

void free_host_int( int* p )
{
  Assert( p );
  free( (void*) p );
}

/*---------------------------------------------------------------------------*/

void free_host_P( P* p )
{
  Assert( p );
  free( (void*) p );
}

/*---------------------------------------------------------------------------*/

void free_host_pinned_P( P* p )
{
  Assert( p );
#ifdef USE_CUDA
  cudaFreeHost( p );
  Assert( Env_cuda_last_call_succeeded() );
#else
  free( (void*) p );
#endif
}

/*---------------------------------------------------------------------------*/

void free_device_P( P* p )
{
#ifdef USE_CUDA
  cudaFree( p );
  Assert( Env_cuda_last_call_succeeded() );
#endif
}

#endif /*---__MIC__---*/

/*---------------------------------------------------------------------------*/

void cuda_copy_host_to_device_P( P*     p_d,
                                 P*     p_h,
                                 size_t n )
{
#ifdef USE_CUDA
  Assert( p_d );
  Assert( p_h );
  Assert( n+1 >= 1 );

  cudaMemcpy( p_d, p_h, n*sizeof(P), cudaMemcpyHostToDevice );
  Assert( Env_cuda_last_call_succeeded() );
#endif
}

/*---------------------------------------------------------------------------*/

void cuda_copy_device_to_host_P( P*     p_h,
                                 P*     p_d,
                                 size_t n )
{
#ifdef USE_CUDA
  Assert( p_h );
  Assert( p_d );
  Assert( n+1 >= 1 );

  cudaMemcpy( p_h, p_d, n*sizeof(P), cudaMemcpyDeviceToHost );
  Assert( Env_cuda_last_call_succeeded() );
#endif
}

/*---------------------------------------------------------------------------*/

void cuda_copy_host_to_device_stream_P( P*       p_d,
                                        P*       p_h,
                                        size_t   n,
                                        Stream_t stream )
{
#ifdef USE_CUDA
  Assert( p_d );
  Assert( p_h );
  Assert( n+1 >= 1 );

  cudaMemcpyAsync( p_d, p_h, n*sizeof(P), cudaMemcpyHostToDevice, stream );
  Assert( Env_cuda_last_call_succeeded() );
#endif
}

/*---------------------------------------------------------------------------*/

void cuda_copy_device_to_host_stream_P( P*       p_h,
                                        P*       p_d,
                                        size_t   n,
                                        Stream_t stream )
{
#ifdef USE_CUDA
  Assert( p_h );
  Assert( p_d );
  Assert( n+1 >= 1 );

  cudaMemcpyAsync( p_h, p_d, n*sizeof(P), cudaMemcpyDeviceToHost, stream );
  Assert( Env_cuda_last_call_succeeded() );
#endif
}

/*===========================================================================*/
/*---Stream management---*/

Stream_t Env_cuda_stream_send_block( Env* env )
{
#ifdef USE_CUDA
  return env->stream_send_block_;
#else
  return 0;
#endif
}

/*---------------------------------------------------------------------------*/

Stream_t Env_cuda_stream_recv_block( Env* env )
{
#ifdef USE_CUDA
  return env->stream_recv_block_;
#else
  return 0;
#endif
}

/*---------------------------------------------------------------------------*/

Stream_t Env_cuda_stream_kernel_faces( Env* env )
{
#ifdef USE_CUDA
  return env->stream_kernel_faces_;
#else
  return 0;
#endif
}

/*---------------------------------------------------------------------------*/

void Env_cuda_stream_wait( Env* env, Stream_t stream )
{
#ifdef USE_CUDA
  cudaStreamSynchronize( stream );
  Assert( Env_cuda_last_call_succeeded() );
#endif
}

/*===========================================================================*/

#ifdef __cplusplus
} /*---extern "C"---*/
#endif

/*---------------------------------------------------------------------------*/
