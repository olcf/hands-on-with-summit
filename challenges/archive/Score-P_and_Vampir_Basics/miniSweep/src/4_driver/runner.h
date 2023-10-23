/*---------------------------------------------------------------------------*/
/*!
 * \file   runner.h
 * \author Wayne Joubert
 * \date   Wed Jan 28 10:11:10 EST 2015
 * \brief  Declarations for tools to perform runs of sweeper, header.
 * \note   Copyright (C) 2013 Oak Ridge National Laboratory, UT-Battelle, LLC.
 */
/*---------------------------------------------------------------------------*/

#ifndef _runner_h_
#define _runner_h_

#include "arguments.h"
#include "env.h"
#include "definitions.h"

#ifdef __cplusplus
extern "C"
{
#endif

/*===========================================================================*/
/*---Struct to hold run result data---*/

typedef struct
{
  P      normsq;
  P      normsqdiff;
  double flops;
  double floprate;
  Timer  time;
} Runner;

/*===========================================================================*/
/*---Null object---*/

Runner Runner_null(void);

/*===========================================================================*/
/*---Pseudo-constructor---*/

void Runner_create( Runner* runner );

/*===========================================================================*/
/*---Pseudo-destructor---*/

void Runner_destroy( Runner* runner );

/*===========================================================================*/
/*---Perform run---*/

void Runner_run_case( Runner* runner, Arguments* args, Env* env );

/*===========================================================================*/
/*---Utility function: perform two runs, compare results---*/

Bool_t compare_runs( const char* argstring1, const char* argstring2, Env* env );

/*===========================================================================*/

#ifdef __cplusplus
} /*---extern "C"---*/
#endif

#endif /*---_runner_h_---*/

/*---------------------------------------------------------------------------*/
