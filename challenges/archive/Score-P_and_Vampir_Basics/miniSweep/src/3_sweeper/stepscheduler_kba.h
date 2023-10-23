/*---------------------------------------------------------------------------*/
/*!
 * \file   stepscheduler_kba.h
 * \author Wayne Joubert
 * \date   Tue Jan 28 16:37:41 EST 2014
 * \brief  Declarations for managing sweep step schedule.
 * \note   Copyright (C) 2014 Oak Ridge National Laboratory, UT-Battelle, LLC.
 */
/*---------------------------------------------------------------------------*/

#ifndef _stepscheduler_kba_h_
#define _stepscheduler_kba_h_

#include "env.h"

#include "stepscheduler_kba_kernels.h"

#ifdef __cplusplus
extern "C"
{
#endif

/*===========================================================================*/
/*---Struct with info to define the sweep step schedule---*/

typedef struct
{
  int nblock_z_;
  int nproc_x_;
  int nproc_y_;
  int nblock_octant_;
  int noctant_per_block_;
} StepScheduler;

/*===========================================================================*/
/*---Null object---*/

StepScheduler StepScheduler_null(void);

/*===========================================================================*/
/*---Pseudo-constructor for StepScheduler struct---*/

void StepScheduler_create( StepScheduler* stepscheduler,
                           int            nblock_z,
                           int            nblock_octant,
                           Env*           env );

/*===========================================================================*/
/*---Pseudo-destructor for StepScheduler struct---*/

void StepScheduler_destroy( StepScheduler* stepscheduler );

/*===========================================================================*/
/*---Accessor: blocks along z axis---*/

int StepScheduler_nblock_z( const StepScheduler* stepscheduler );

/*===========================================================================*/
/*---Number of block steps executed for a single octant in isolation---*/

int StepScheduler_nblock( const StepScheduler* stepscheduler );

/*===========================================================================*/
/*---Number of octants per octant block---*/

int StepScheduler_noctant_per_block( const StepScheduler* stepscheduler );

/*===========================================================================*/
/*---Number of kba parallel steps---*/

int StepScheduler_nstep( const StepScheduler* stepscheduler );

/*===========================================================================*/
/*---Get information describing a sweep step---*/

StepInfo StepScheduler_stepinfo( const StepScheduler* stepscheduler,  
                                 const int            step,
                                 const int            octant_in_block,
                                 const int            proc_x,
                                 const int            proc_y );

/*===========================================================================*/
/*---Determine whether to send a face computed at step, used at step+1---*/

Bool_t StepScheduler_must_do_send(
  StepScheduler* stepscheduler,
  int            step,
  int            axis,
  int            dir_ind,
  int            octant_in_block,
  Env*           env );

/*===========================================================================*/
/*---Determine whether to recv a face computed at step, used at step+1---*/

Bool_t StepScheduler_must_do_recv(
  StepScheduler* stepscheduler,
  int            step,
  int            axis,
  int            dir_ind,
  int            octant_in_block,
  Env*           env );

/*===========================================================================*/

#ifdef __cplusplus
} /*---extern "C"---*/
#endif

#endif /*---_stepscheduler_kba_h_---*/

/*---------------------------------------------------------------------------*/
