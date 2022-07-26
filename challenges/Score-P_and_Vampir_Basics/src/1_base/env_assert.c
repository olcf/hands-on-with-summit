/*---------------------------------------------------------------------------*/
/*!
 * \file   env_assert.c
 * \author Wayne Joubert
 * \date   Wed Jan 15 16:06:28 EST 2014
 * \brief  Environment settings for assertions.
 * \note   Copyright (C) 2014 Oak Ridge National Laboratory, UT-Battelle, LLC.
 */
/*---------------------------------------------------------------------------*/

#include "env_assert.h"

#ifndef __CUDA_ARCH__
/*---Do the following on the HDST---*/

#include <stdlib.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C"
{
#endif

/*===========================================================================*/
/*---Assertions---*/

void insist_( const char *condition_string, const char *file, int line )
{
  fprintf( stderr, "Insist error: \"%s\". At file %s, line %i.\n",
                   condition_string, file, line );
  exit( EXIT_FAILURE );
}

/*===========================================================================*/

#ifdef __cplusplus
} /*---extern "C"---*/
#endif

#endif /*---__CUDA_ARCH__---*/

/*---------------------------------------------------------------------------*/
