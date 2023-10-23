/*---------------------------------------------------------------------------*/
/*!
 * \file   env.h
 * \author Wayne Joubert
 * \date   Wed Jan 15 16:06:28 EST 2014
 * \brief  Environment settings specific to programming APIs, header.
 * \note   Copyright (C) 2014 Oak Ridge National Laboratory, UT-Battelle, LLC.
 */
/*---------------------------------------------------------------------------*/

/*=============================================================================

This is the main header file providing access to the specific parallel
environemnt being used.  Other header files are included for the candidate APIs
which are activated if being used.

=============================================================================*/

#ifndef _env_h_
#define _env_h_

#include "arguments.h"

/*---Header file for assertions---*/
#include "env_assert.h"

/*---Data structure declarations relevant to env---*/
#include "env_types.h"

/*---Definitions relevant to specific parallel APIs---*/
#include "env_mpi.h"
#include "env_openmp.h"
#include "env_cuda.h"
#include "env_mic.h"

/*===========================================================================*/

#ifdef __cplusplus
extern "C"
{
#endif

/*===========================================================================*/
/*---Null object---*/

Env Env_null(void);

/*===========================================================================*/
/*---Initialize for execution---*/

void Env_initialize( Env *env, int argc, char** argv );

/*===========================================================================*/
/*---Set values from args---*/

void Env_set_values( Env *env, Arguments* args );

/*===========================================================================*/
/*---Finalize execution---*/

void Env_finalize( Env* env );

/*===========================================================================*/
/*---Indicate whether to do output---*/

Bool_t Env_is_proc_master( Env* env );

/*===========================================================================*/
/*---Timer type---*/

typedef double Timer;

/*===========================================================================*/
/*---Timer utilities---*/

Timer Env_get_time( Env* env );

/*---------------------------------------------------------------------------*/

Timer Env_get_synced_time( Env* env );

/*===========================================================================*/

#ifdef __cplusplus
} /*---extern "C"---*/
#endif

#endif /*---_env_h_---*/

/*---------------------------------------------------------------------------*/
