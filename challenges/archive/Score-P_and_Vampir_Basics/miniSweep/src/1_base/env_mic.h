/*---------------------------------------------------------------------------*/
/*!
 * \file   env_mic.h
 * \author Wayne Joubert
 * \date   Wed Jun 11 09:33:15 EDT 2014
 * \brief  Environment settings for Intel MIC.
 * \note   Copyright (C) 2014 Oak Ridge National Laboratory, UT-Battelle, LLC.
 */
/*---------------------------------------------------------------------------*/

#ifndef _env_mic_h_
#define _env_mic_h_

#include <stddef.h>

#include "types.h"
#include "env_mic_kernels.h"

/*===========================================================================*/
/*---Memory management---*/

#ifdef __MIC__

static int* malloc_host_int( size_t n )
{
  Assert( n+1 >= 1 );
  int* result = (int*)malloc( n * sizeof(int) );
  Assert( result );
  return result;
}

/*---------------------------------------------------------------------------*/

static P* malloc_host_P( size_t n )
{
  Assert( n+1 >= 1 );
  P* result = _mm_malloc( n * sizeof(P), VEC_LEN * sizeof(P) );
  Assert( result );
  return result;
}

/*---------------------------------------------------------------------------*/

static P* malloc_host_pinned_P( size_t n )
{
  return malloc_host_P( n );
}

/*---------------------------------------------------------------------------*/

static P* malloc_device_P( size_t n )
{
  Assert( n+1 >= 1 );
  P* result = NULL;
  return result;
}

/*---------------------------------------------------------------------------*/

static void free_host_int( int* p )
{
  Assert( p );
  free( (void*) p );
}

/*---------------------------------------------------------------------------*/

static void free_host_P( P* p )
{
  Assert( p );
  _mm_free( p );
}

/*---------------------------------------------------------------------------*/

static void free_host_pinned_P( P* p )
{
  free_host_P( p );
}

/*---------------------------------------------------------------------------*/

static void free_device_P( P* p )
{
}

#endif /*---__MIC__---*/

/*===========================================================================*/

#endif /*---_env_mic_h_---*/

/*---------------------------------------------------------------------------*/
