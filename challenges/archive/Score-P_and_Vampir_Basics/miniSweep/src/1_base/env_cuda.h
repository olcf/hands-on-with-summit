/*---------------------------------------------------------------------------*/
/*!
 * \file   env_cuda.h
 * \author Wayne Joubert
 * \date   Tue Apr 22 17:03:08 EDT 2014
 * \brief  Environment settings for cuda, header.
 * \note   Copyright (C) 2014 Oak Ridge National Laboratory, UT-Battelle, LLC.
 */
/*---------------------------------------------------------------------------*/

#ifndef _env_cuda_h_
#define _env_cuda_h_

#include <stddef.h>
#include <stdlib.h>

#ifdef USE_CUDA
#include "cuda.h"
#endif

#include "types.h"
#include "env_cuda_kernels.h"

#ifdef __cplusplus
extern "C"
{
#endif

/*===========================================================================*/
/*---Error handling---*/

Bool_t Env_cuda_last_call_succeeded(void);

/*===========================================================================*/
/*---Initialize CUDA---*/

void Env_cuda_initialize_( Env *env, int argc, char** argv );

/*===========================================================================*/
/*---Finalize CUDA---*/

void Env_cuda_finalize_( Env* env );

/*===========================================================================*/
/*---Set values from args---*/

void Env_cuda_set_values_( Env *env, Arguments* args );

/*===========================================================================*/
/*---Determine whether using device---*/

Bool_t Env_cuda_is_using_device( const Env* const env );

/*===========================================================================*/
/*---Memory management, for CUDA and all platforms ex. MIC---*/

#ifndef __MIC__

int* malloc_host_int( size_t n );

/*---------------------------------------------------------------------------*/

P* malloc_host_P( size_t n );

/*---------------------------------------------------------------------------*/

P* malloc_host_pinned_P( size_t n );

/*---------------------------------------------------------------------------*/

P* malloc_device_P( size_t n );

/*---------------------------------------------------------------------------*/

void free_host_int( int* p );

/*---------------------------------------------------------------------------*/

void free_host_P( P* p );

/*---------------------------------------------------------------------------*/

void free_host_pinned_P( P* p );

/*---------------------------------------------------------------------------*/

void free_device_P( P* p );

#endif /*---__MIC__---*/

/*---------------------------------------------------------------------------*/

void cuda_copy_host_to_device_P( P*     p_d,
                                 P*     p_h,
                                 size_t n );

/*---------------------------------------------------------------------------*/

void cuda_copy_device_to_host_P( P*     p_h,
                                 P*     p_d,
                                 size_t n );

/*---------------------------------------------------------------------------*/

void cuda_copy_host_to_device_stream_P( P*       p_d,
                                        P*       p_h,
                                        size_t   n,
                                        Stream_t stream );

/*---------------------------------------------------------------------------*/

void cuda_copy_device_to_host_stream_P( P*       p_h,
                                        P*       p_d,
                                        size_t   n,
                                        Stream_t stream );

/*===========================================================================*/
/*---Stream management---*/

Stream_t Env_cuda_stream_send_block( Env* env );

/*---------------------------------------------------------------------------*/

Stream_t Env_cuda_stream_recv_block( Env* env );

/*---------------------------------------------------------------------------*/

Stream_t Env_cuda_stream_kernel_faces( Env* env );

/*---------------------------------------------------------------------------*/

void Env_cuda_stream_wait( Env* env, Stream_t stream );

/*===========================================================================*/

#ifdef __cplusplus
} /*---extern "C"---*/
#endif

#endif /*---_env_cuda_h_---*/

/*---------------------------------------------------------------------------*/
